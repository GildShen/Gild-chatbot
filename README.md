# Gild Chatbot: AI Agent Classroom Demo

This repository is a small classroom demo project for introducing AI agents. It uses Streamlit as a simple web interface and AutoGen to demonstrate several common agent patterns.

## What This Project Demonstrates

- **Single-agent interaction**: the user asks a question, and one agent responds.
- **Tool-using agent**: the agent can call functions such as checking the time, searching news, finding experts, and finding textbooks.
- **Two-agent conversation**: a student agent and a teacher agent interact with each other.
- **Group-agent conversation**: multiple agents, such as a teacher, tech support agent, general support agent, and student, participate in a shared discussion.

## Teaching Goals

This project is mainly useful for teaching:

- The difference between a chatbot and an agent
- How system prompts and personas shape agent behavior
- How tool calling and function calling work
- The basic idea of multi-agent collaboration
- How Streamlit can be used to quickly build an agent demo interface

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

The app expects API keys to be available through environment variables. You can place them in a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Run the App

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in the terminal.

## Project Structure

- `streamlit_app.py`: main chatbot demo page
- `pages/one_agent.py`: single teacher-agent demo with tool usage
- `pages/two_agents.py`: two-agent conversation demo
- `pages/group_agents.py`: group-agent conversation demo
- `coding/constant.py`: sample task definitions, expert data, and textbook data
- `coding/agenttools.py`: tool functions used by the agents
- `coding/utils.py`: Streamlit helper functions for navigation, chat display, and chat-log saving
