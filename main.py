"""
    Main APP and UI
"""
# pylint: disable=C0301,C0103,C0303,C0411,C0304,W1203

import logging
import streamlit as st

from utils_streamlit import streamlit_hack_remove_top_space, hide_footer, create_sticky_header
from utils.app_logger import init_streamlit_logger
from ui.dialog_storage import DialogStorage, DialogRole
from backend.backend import Backend

init_streamlit_logger()

logger : logging.Logger = logging.getLogger()

# ------------------------------- Session

SESSION_BACKEND = 'backend_session'
if SESSION_BACKEND not in st.session_state:
    st.session_state[SESSION_BACKEND] = Backend()
backend = st.session_state[SESSION_BACKEND]

SESSION_DIALOG_STORAGE = 'dialog_storage'
if SESSION_DIALOG_STORAGE not in st.session_state:
    st.session_state[SESSION_DIALOG_STORAGE] = DialogStorage()
dialog_storage = st.session_state[SESSION_DIALOG_STORAGE]

SESSION_TOKENS_COUNT = 'tokens_count'
if SESSION_TOKENS_COUNT not in st.session_state:
    st.session_state[SESSION_TOKENS_COUNT] = 0

SESSION_QUERY_EXAMPLES = 'query_examples'
if SESSION_QUERY_EXAMPLES not in st.session_state:
    st.session_state[SESSION_QUERY_EXAMPLES] = []

# ------------------------------- UI
st.set_page_config(page_title= "Demo POC", layout="wide")

streamlit_hack_remove_top_space()
hide_footer()

header = st.container()
create_sticky_header(header, "Demo")

progress = st.empty()

with st.sidebar:
    with st.form(key="uploadContent", clear_on_submit=True):
        uploaded_index_file = st.file_uploader(
            'Upload chat index',
            type=["zip"],
            accept_multiple_files= False,
            key="uploaded_index_zip"
        )
        load_button = st.form_submit_button(label="Upload")
    create_summary = st.checkbox(label="Create summary (LLM)")
    ignore_score_threshold = st.checkbox(label="Ignore score threshold")
    enable_step_back = st.checkbox(label="Enable step back")
    query_examples = st.expander(label="Query examples", expanded= True)
    tokens_used_container = st.empty()

def refresh_tokens_used():
    """Refresh tokens used"""
    tokens_used_container.markdown(f'Tokens used {st.session_state[SESSION_TOKENS_COUNT]}')

def exec_system_prompt(command : str):
    """Run system prompt"""
    if command == "/clear":
        dialog_storage.clear()

refresh_tokens_used()

if not st.session_state[SESSION_QUERY_EXAMPLES]:
    st.session_state[SESSION_QUERY_EXAMPLES] = backend.get_query_examples()
query_examples.markdown('\n'.join(st.session_state[SESSION_QUERY_EXAMPLES]))

if load_button:
    if not uploaded_index_file:
        progress.markdown('Please add index file to upload')
        st.stop()
    backend.upload_index(uploaded_index_file)

prompt = st.chat_input("Your question or command")
if prompt:
    if str(prompt).startswith("/"):
        exec_system_prompt(prompt)
        st.stop()
    dialog_storage.add_message(DialogRole.USER, prompt)

    search_result = backend.get_chunks(prompt, ignore_score_threshold)
    if not search_result and enable_step_back:
        logger.info("Run step back strategy")
        step_back_answer = backend.get_step_back(prompt)
        logger.info(f"Step back prompt {step_back_answer}")
        if not step_back_answer.error:
            search_result = backend.get_chunks(step_back_answer.answer, ignore_score_threshold)

    if search_result:
        if not create_summary:
            for s in search_result:
                content_str = f'[{s.score:.02}] {s.content}'
                dialog_storage.add_message(DialogRole.ASSISTANT, content_str)
        else:
            answer = backend.create_summary(prompt, search_result)
            if not answer.error:
                dialog_storage.add_message(DialogRole.ASSISTANT, answer.answer)
            else:
                dialog_storage.add_message(DialogRole.ASSISTANT, answer.error)
            st.session_state[SESSION_TOKENS_COUNT] += answer.tokens_used
            refresh_tokens_used()
    else:
        dialog_storage.add_message(DialogRole.ASSISTANT, "Sorry, I have no answer to your question")

dialog_items = dialog_storage.get_message_list()
for dialog_item in dialog_items:
    with st.chat_message(dialog_item.role.value):
        st.write(dialog_item.msg)
