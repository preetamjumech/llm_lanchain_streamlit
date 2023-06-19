import os
from apikey import apikey


import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain # for the cinnection between multiple outputs
from langchain.memory import ConversationBufferMemory # for memory
from langchain.utilities import WikipediaAPIWrapper
import time
time.clock = time.time

os.environ["OPENAI_API_KEY"] = apikey
st.title('🦜️🔗 Youtube GPT Creator')
prompt = st.text_input("Plug in your prompt here")

#Prompt templates
title_template = PromptTemplate(
    input_variables = ["topic"],
    template = "Write me a youtube video title {topic}"
)

script_template = PromptTemplate(
    input_variables = ["title", "wikipedia_research"],
    template = "Write me a youtube video  script based on this title TITLE: {title} while leveraging this wikipedia research: {wikipedia_research} "
)
# Mermory
title_memory = ConversationBufferMemory(input_key = "topic", memory_key = "chat_history")
script_memory = ConversationBufferMemory(input_key = "title", memory_key = "chat_history")
# LLM
llm = OpenAI(temperature = 0.9)
title_chain = LLMChain(llm = llm,prompt = title_template,verbose = True,output_key = "title", memory = title_memory)
script_chain = LLMChain(llm = llm,prompt = script_template,verbose = True,output_key = "script", memory = script_memory)
wiki = WikipediaAPIWrapper()
# sequential_chain = SequentialChain(chains= [title_chain,script_chain],verbose = True,input_variables = ["topic"],
# output_variables=["title","script"]) 

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title, wikipedia_research = wiki_research)
    # st.write(response["title"])
    # st.write(response["script"])
    st.write(title)
    st.write(script)
    
    with st.expander("Message History"):
        st.info(title_memory.buffer)
    with st.expander("Script History"):
        st.info(script_memory.buffer)
    with st.expander("Wkipedia Research"):
        st.info(wiki_research)