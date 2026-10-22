import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.chat_memory import (
    as_langchain_history,
    ensure_chat_session,
    history_for_prompt,
    max_history_turns,
    needs_query_rewrite,
    reset_chat_session,
    rewrite_standalone_query,
)
from src.search import RAGSearch


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return type("Response", (), {"content": response})()


class FakeVectorStore:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else [
            {
                "distance": 0.1,
                "metadata": {
                    "text": "WAM stores warehouse information in the indexed system.",
                    "source_label": "WAM.pdf",
                    "page": 0,
                },
            }
        ]
        self.queries = []
        self.write_calls = 0
        self.collection = type("Collection", (), {"count": lambda _: 12})()

    def query(self, query, top_k=5):
        self.queries.append((query, top_k))
        return self.rows


class ChatMemoryTests(unittest.TestCase):
    def test_session_history_is_preserved_as_langchain_messages(self):
        state = {}
        ensure_chat_session(state)
        state["chat_history"].extend([
            HumanMessage(content="What is WAM?"),
            AIMessage(content="WAM is the warehouse application."),
        ])
        messages = as_langchain_history(state["chat_history"])
        self.assertEqual([message.type for message in messages], ["human", "ai"])
        self.assertEqual(messages[0].content, "What is WAM?")
        self.assertIn("warehouse application", history_for_prompt(messages))

    def test_follow_up_is_rewritten_using_previous_context(self):
        llm = FakeLLM("What database does WAM use?")
        history = [HumanMessage(content="What is WAM?"), AIMessage(content="WAM is a warehouse application.")]
        rewritten = rewrite_standalone_query(llm, "What database does it use?", history)
        self.assertEqual(rewritten, "What database does WAM use?")
        self.assertTrue(any(message.content == "WAM is a warehouse application." for message in llm.calls[0]))

    def test_standalone_question_is_not_rewritten(self):
        llm = FakeLLM("should not be used")
        history = [HumanMessage(content="What is WAM?"), AIMessage(content="A warehouse application.")]
        question = "What is Vendor Maintenance?"
        self.assertFalse(needs_query_rewrite(question))
        self.assertEqual(rewrite_standalone_query(llm, question, history), question)
        self.assertEqual(llm.calls, [])

    def test_standalone_question_searches_original_query_without_history_prompt(self):
        store = FakeVectorStore()
        llm = FakeLLM("WAM is a warehouse application. [1]")
        rag = object.__new__(RAGSearch)
        rag.vectorstore = store
        rag.llm = llm
        history = [HumanMessage(content="What is a previous project?"), AIMessage(content="Old context.")]
        rag.search_and_summarize("What is WAM?", history=history)
        self.assertEqual(store.queries[0][0], "What is WAM?")
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("Conversation history for reference resolution only", llm.calls[0][0])
        self.assertIn("(none)", llm.calls[0][0])
        self.assertNotIn("Old context", llm.calls[0][0])

    def test_history_is_limited_to_configured_recent_turns(self):
        history = []
        for index in range(8):
            history.extend([
                HumanMessage(content=f"User turn {index}"),
                AIMessage(content=f"Assistant turn {index}"),
            ])
        with patch.dict(os.environ, {"MAX_HISTORY_TURNS": "6"}):
            self.assertEqual(max_history_turns(), 6)
            llm = FakeLLM("What database does WAM use?")
            rewrite_standalone_query(llm, "What database does it use?", history)
        sent = llm.calls[0]
        history_sent = sent[1:-1]
        self.assertEqual(len(history_sent), 12)
        self.assertEqual(history_sent[0].content, "User turn 2")
        self.assertEqual(history_sent[-1].content, "Assistant turn 7")
        self.assertNotIn("User turn 1", [message.content for message in sent])

    def test_memory_is_used_for_rewrite_not_written_to_chromadb(self):
        store = FakeVectorStore()
        llm = FakeLLM(
            "What database does WAM use?",
            "The retrieved document identifies the system. [1]",
        )
        rag = object.__new__(RAGSearch)
        rag.vectorstore = store
        rag.llm = llm
        history = [HumanMessage(content="What is WAM?"), AIMessage(content="WAM is a warehouse application.")]
        result = rag.search_and_summarize("What database does it use?", history=history)
        self.assertEqual(store.queries[0][0], "What database does WAM use?")
        self.assertNotIn("WAM is a warehouse application", store.queries[0][0])
        self.assertEqual(store.write_calls, 0)
        self.assertEqual(result["sources"][0]["document"], "WAM.pdf")
        answer_prompt = llm.calls[-1][0]
        self.assertIn("Conversation history for reference resolution only", answer_prompt)
        self.assertIn("not factual evidence", answer_prompt)
        self.assertIn("different applications that have similar names", answer_prompt)
        self.assertIn("Current user question: What database does it use?", answer_prompt)

    def test_rewrite_failure_falls_back_to_original_query_and_retrieves(self):
        store = FakeVectorStore()
        llm = FakeLLM(RuntimeError("rewriter unavailable"), "Retrieved answer. [1]")
        rag = object.__new__(RAGSearch)
        rag.vectorstore = store
        rag.llm = llm
        history = [HumanMessage(content="What is WAM?"), AIMessage(content="A warehouse application.")]
        result = rag.search_and_summarize("Who uses it?", history=history)
        self.assertEqual(store.queries[0][0], "Who uses it?")
        self.assertEqual(result["answer"], "Retrieved answer. [1]")

    def test_new_chat_resets_messages_memory_and_session_id_only(self):
        state = {}
        ensure_chat_session(state)
        old_id = state["chat_session_id"]
        old_history = state["chat_history"]
        state["chat_messages"].append({"role": "assistant", "content": "Old answer"})
        old_history.append(HumanMessage(content="Old question"))
        state["view_source_0_1_open"] = True
        state["active_page"] = "Documents"
        store = FakeVectorStore()
        indexed_count_before = store.collection.count()

        reset_chat_session(state)

        self.assertEqual(state["chat_messages"], [])
        self.assertEqual(state["chat_history"], [])
        self.assertNotEqual(state["chat_session_id"], old_id)
        self.assertIsNot(state["chat_history"], old_history)
        self.assertEqual(state["active_page"], "Chat")
        self.assertNotIn("view_source_0_1_open", state)
        self.assertEqual(store.collection.count(), indexed_count_before)

    def test_empty_retrieval_keeps_safe_no_answer_without_sources(self):
        store = FakeVectorStore(rows=[])
        llm = FakeLLM("should not be called")
        rag = object.__new__(RAGSearch)
        rag.vectorstore = store
        rag.llm = llm
        result = rag.search_and_summarize("Unsupported question")
        self.assertEqual(result["answer"], RAGSearch._no_answer())
        self.assertEqual(result["sources"], [])
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
