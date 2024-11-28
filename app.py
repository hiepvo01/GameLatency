import streamlit as st

pages = {
    "Home": [ 
        st.Page("pages/landing_page.py", title="Welcome", icon=":material/waving_hand:"),
    ],
    "Demographic": [
        st.Page("pages/demographics.py", title="Participant Analysis", icon=":material/demography:"),  
    ],
    "Reports": [
        st.Page("pages/visualizations/strategy_distribution.py", title="Player Strategy Distribution", icon=":material/grid_on:"),
        st.Page("pages/visualizations/latency_impact_strategy.py", title="Latency Impact Strategy Adaptation", icon=":material/grid_on:"),
        st.Page("pages/visualizations/latency_impact_weapon_kill.py", title="Latency Impact Killing Weapon", icon=":material/grid_on:"),
    ],
    "Gaming Experiments":[
        st.Page("pages/utsexperiments.py", title="UTS Campus Experiments", icon=":material/person:"),
        st.Page("pages/eventexperiments.py", title="Large Event Experiments", icon=":material/person:"),
    ],
    "Support": [
        st.Page("pages/how_to.py", title="How to", icon=":material/info:"),
    ],   
}

pg = st.navigation(pages)
pg.run()