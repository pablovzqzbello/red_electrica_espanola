import streamlit as st

def configure_page(
    title="PFB: Red Eléctrica Española",
    icon=":bulb:",
    layout="wide",
    initial_sidebar_state="auto"
):
    """
    Parameters:
    - title (str): Título de la página que aparece en el navegador.
    - icon (str): Emoji o icono que aparece en la pestaña del navegador.
    - layout (str): Disposición de la página ("centered" o "wide").
    - initial_sidebar_state (str): Estado inicial del sidebar ("auto", "expanded", "collapsed").
    """

    if not hasattr(st, '_configured_page'):
        st.set_page_config(
            page_title=title,
            page_icon=icon,
            layout=layout,
            initial_sidebar_state=initial_sidebar_state
        )
        st._configured_page = True

## Con esto prevenimos el mensaje de error si intentasemos invocar la función configure_page más de una vez