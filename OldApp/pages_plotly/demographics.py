import streamlit as st
import pandas as pd
import plotly.express as px
import textwrap

st.set_page_config(page_title="Demographics Analysis", page_icon="📊", layout="wide")

# File path
csv_file = 'survey-data/demographics.csv'

# Function to plot interactive pie chart for a categorical column
def plot_interactive_pie_chart(column_name, df):
    # Count unique values for the column, excluding NaN
    counts = df[column_name].value_counts(dropna=True)
    
    # Wrap the title if it's too long
    wrapped_title = '<br>'.join(textwrap.wrap(f'Distribution of {column_name}', width=50))
    
    # Create an interactive pie chart using Plotly
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title=wrapped_title,
        hover_data=[counts.values],  # Show the actual count on hover
        labels={'label': column_name, 'value': 'Count'}
    )
    
    # Update layout for better appearance
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=False,
        height=600,  # Slightly reduced height
        width=800,
        title={
            'text': f'<b>{wrapped_title}</b>',
            'y':0.95,  # Slightly lowered the title
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18}
        },
        margin=dict(t=60, b=20, l=20, r=20)  # Reduced top margin
    )
    
    return fig

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(csv_file)

# Main content
st.title('Interactive Participant Demographics Analysis')
st.info('Interactive Pie Charts of Demographics Data')

# Load the data
df = load_data()

if df is not None:
    # Display column selection and pie charts
    columns_to_plot = st.multiselect(
        'Select Columns to Plot', 
        df.columns, 
        default="What platforms do you use to play games? (e.g., PC, console, mobile)"
    )
    
    if columns_to_plot:
        for column in columns_to_plot:
            fig = plot_interactive_pie_chart(column, df)
            # Use columns to center the chart
            col1, col2, col3 = st.columns([1,6,1])
            with col2:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one column to display the chart.")
else:
    st.error("Failed to load the data. Please check the file path and try again.")