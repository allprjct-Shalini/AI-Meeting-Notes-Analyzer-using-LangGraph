import streamlit as st
import os
from typing import TypedDict, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

st.set_page_config(page_title="AI Meeting Notes Analyzer", page_icon="📊", layout="wide")

st.title("📊 AI Meeting Notes Analyzer using LangGraph")
st.write("Convert raw meeting transcripts into structured summaries, action items, and priority levels automatically.")

# Sidebar for API Key
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Transcript Input
transcript_input = st.text_area(
    "Paste Meeting Transcript Here:", 
    height=200,
    value="John: We need to improve the website performance.\nSarah: Yes, page load time is too slow.\nDavid: I will optimize the database queries this week.\nSarah: I will redesign the homepage layout.\nJohn: Let's try to finish these tasks before Friday."
)

if st.button("Analyze Meeting", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not transcript_input.strip():
        st.error("Please enter a valid meeting transcript.")
    else:
        with st.spinner("Running LangGraph workflow agents..."):
            os.environ["OPENAI_API_KEY"] = api_key
            llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
            
            # Define State
            class MeetingState(TypedDict):
                transcript: str
                topics: List[str]
                summary: str
                action_items: List[str]
                has_action_items: bool
                priority: str
                final_report: str

            # Nodes
            def topic_extraction_node(state: MeetingState):
                prompt = f"Analyze the following meeting transcript and identify the main discussion topics. Return as a bulleted list.\n\nTranscript:\n{state['transcript']}"
                res = llm.invoke([HumanMessage(content=prompt)])
                topics = [t.strip("- *") for t in res.content.strip().split("\n") if t.strip()]
                return {"topics": topics}

            def summary_node(state: MeetingState):
                prompt = f"Generate a concise meeting summary (3-5 sentences).\n\nTranscript:\n{state['transcript']}"
                res = llm.invoke([HumanMessage(content=prompt)])
                return {"summary": res.content}

            def action_item_node(state: MeetingState):
                prompt = f"Identify tasks and owners (label 'Owner: Not specified' if missing). If none, state 'No action items'.\n\nTranscript:\n{state['transcript']}"
                res = llm.invoke([HumanMessage(content=prompt)])
                content = res.content
                has_items = "no action items" not in content.lower()
                items = [i.strip("- *") for i in content.strip().split("\n") if i.strip()] if has_items else []
                return {"action_items": items, "has_action_items": has_items}

            def priority_node(state: MeetingState):
                prompt = f"Determine priority level ('High Priority', 'Medium Priority', or 'Low Priority') based on urgency phrases.\n\nTranscript:\n{state['transcript']}"
                res = llm.invoke([HumanMessage(content=prompt)])
                return {"priority": res.content.strip()}

            def skip_priority_node(state: MeetingState):
                return {"priority": "No action items identified in this meeting."}

            def final_output_node(state: MeetingState):
                topics_str = "\n".join([f"- {t}" for t in state.get('topics', [])])
                actions_str = "\n".join([f"- {a}" for a in state.get('action_items', [])]) if state.get('has_action_items') else "No action items identified."
                report = f"""### 📝 Meeting Summary\n{state.get('summary', 'N/A')}\n\n### 🔑 Key Topics\n{topics_str}\n\n### ✅ Action Items & Owners\n{actions_str}\n\n### ⚡ Priority Level\n{state.get('priority', 'N/A')}"""
                return {"final_report": report}

            def check_action_items(state: MeetingState):
                return "priority_node" if state.get("has_action_items", False) else "skip_priority_node"

            # Build Graph
            workflow = StateGraph(MeetingState)
            workflow.add_node("topic_extraction", topic_extraction_node)
            workflow.add_node("summary", summary_node)
            workflow.add_node("action_items", action_item_node)
            workflow.add_node("priority_node", priority_node)
            workflow.add_node("skip_priority_node", skip_priority_node)
            workflow.add_node("final_output", final_output_node)

            workflow.add_edge(START, "topic_extraction")
            workflow.add_edge("topic_extraction", "summary")
            workflow.add_edge("summary", "action_items")
            workflow.add_conditional_edges("action_items", check_action_items, {"priority_node": "priority_node", "skip_priority_node": "skip_priority_node"})
            workflow.add_edge("priority_node", "final_output")
            workflow.add_edge("skip_priority_node", "final_output")
            workflow.add_edge("final_output", END)

            app = workflow.compile()
            result = app.invoke({"transcript": transcript_input, "topics": [], "summary": "", "action_items": [], "has_action_items": False, "priority": "", "final_report": ""})
            
            st.success("Analysis Complete!")
            st.markdown(result["final_report"])