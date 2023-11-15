"""
    Utils for streamlit
"""

import streamlit as st
import streamlit.components.v1 as components

# https://discuss.streamlit.io/t/prevent-st-text-input-from-triggering-callback-when-losing-focus/37103/3

def streamlit_hack_disable_textarea_submit():
    """Do not submit TextArea component when lost focus"""
    components.html(
            """
        <script>
        const doc = window.parent.document;

        const textareas = doc.querySelectorAll('textarea');
        textareas.forEach(textarea => {
            textarea.addEventListener('focusout', function(event) {
                event.stopPropagation();
                event.preventDefault();
            });
        });

        const inputs = doc.querySelectorAll('input');
        inputs.forEach(input => {
            input.addEventListener('focusout', function(event) {
                event.stopPropagation();
                event.preventDefault();
            });
        });
        
        </script>""",
            height=0,
            width=0,
        )

def streamlit_hack_remove_top_space():
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                }
               .css-1544g2n  {
                    padding-top: 2rem;
                }
                .css-10oheav {
                    padding-top: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

def hide_footer():
    """Hide footer"""
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def create_sticky_header(container, header : str):
    """Create sticky header"""
    container.title(header)
    container.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    st.markdown(
        """
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
                .fixed-header {
                    border-bottom: 1px solid black;
                }
            </style>
        """,
        unsafe_allow_html=True
    )
