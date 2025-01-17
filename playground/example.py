from langchain_writer import ChatWriter


def invoke():
    chat_writer = ChatWriter()
    response = chat_writer.invoke("Hello")

    print(response)


invoke()
