import pytest

from langchain_writer import ChatWriter, GraphTool, WriterTextSplitter
from tests.integration_tests.test_tools import GRAPH_IDS


@pytest.fixture(scope="function")
def chat_writer():
    return ChatWriter()


@pytest.fixture(scope="function")
def graph_tool():
    return GraphTool(graph_ids=GRAPH_IDS)


@pytest.fixture(scope="function")
def text_splitter():
    return WriterTextSplitter()


@pytest.fixture(scope="function")
def text_to_split():
    return """
    Writer AI module

    This module leverages the Writer Python SDK to enable applications to interact with large language models (LLMs)
    in chat or text completion formats. It provides tools to manage conversation states and to dynamically interact
    with LLMs using both synchronous and asynchronous methods.

    Getting your API key

    To utilize the Writer AI module, you’ll need to configure the WRITER_API_KEY environment variable with an API
    key obtained from AI Studio. Here is a detailed guide to setup up this key. You will need to select an API app
    under Developer tools

    Once you have your API key, set it as an environment variable on your system:

    export WRITER_API_KEY=your_api_key_here

    You can manage your environment variables using methods that best suit your setup,
    such as employing tools like python-dotenv.

    Furthermore, when deploying an application with writer deploy,
    the WRITER_API_KEY environment variable is automatically configured with the API key
    specified during the deployment process.

    Chat completion with the Conversation class

    The Conversation class manages LLM communications within a chat framework,
    storing the conversation history and handling the interactions.

    <code>
        import writer as wf
        import writer.ai

        def handle_simple_message(state, payload):
            # Update the conversation state by appending the incoming user message.
            state["conversation"] += payload

            # Stream the complete response from the AI model in chunks.
            for chunk in state["conversation"].stream_complete():
                # Append each chunk of the model's response to the ongoing conversation state.
                state["conversation"] += chunk

        # Initialize the application state with a new Conversation object.
        initial_state = wf.init_state({
            "conversation": writer.ai.Conversation(),
        })
    </code>
    """
