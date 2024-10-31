import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os 
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER, IMPORT_FOLDER, RULES_PATH

def show_demographic():

    csv_file = f'{IMPORT_FOLDER}/demographics.csv'
    insight_rules_file = f'{RULES_PATH}/insight_rules.csv'

    def preprocess_data(df):
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].str.split(',')
        return df

    def load_insight_rules():
        if not os.path.exists(insight_rules_file):
            st.error(f"Insight rules file not found: {insight_rules_file}")
            return None
        try:
            return pd.read_csv(insight_rules_file)
        except Exception as e:
            st.error(f"Error loading insight rules: {str(e)}")
            return None

    def get_insight(question, value, rules_df):
        if rules_df is None:
            return "No insights available."
        
        matching_rules = rules_df[rules_df['Question'] == question]
        
        if matching_rules.empty:
            return f"No specific insight available for {question}."

        for _, rule in matching_rules.iterrows():
            condition = rule['Condition']
            
            # Handle different types of conditions
            if isinstance(value, list):  # Multiple choice
                if any(v.strip() in condition for v in value):
                    return rule['Insight']
            elif '-' in condition:  # Range
                try:
                    low, high = map(int, condition.split('-'))
                    if low <= float(value) <= high:
                        return rule['Insight']
                except ValueError:
                    pass
            elif condition.startswith('>='):  # Greater than or equal
                try:
                    if float(value) >= float(condition[2:]):
                        return rule['Insight']
                except ValueError:
                    pass
            elif str(value).lower() == condition.lower():  # Exact match (case-insensitive)
                return rule['Insight']
        
        return f"No specific insight available for {value} in {question}."

    def plot_overall_chart(df, column_name):
        if isinstance(df[column_name].iloc[0], list):
            # Multiple choice question
            temp_df = df[column_name].explode()
            counts = temp_df.value_counts(dropna=True)
        else:
            # Single choice question
            counts = df[column_name].value_counts(dropna=True)
        
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            title=f'Distribution of {column_name}',
            labels={'x': 'Response', 'y': 'Count'},
        )
        
        fig.update_layout(xaxis_tickangle=-45, height=400, width=600)
        
        # Calculate percentage for each category
        total = counts.sum()
        percentages = (counts / total * 100).round(1)
        
        # Add percentage labels on top of each bar
        for i, (value, percentage) in enumerate(zip(counts, percentages)):
            fig.add_annotation(
                x=counts.index[i],
                y=value,
                text=f"{percentage}%",
                showarrow=False,
                yshift=10
            )

        # Determine the Majority
        dominant_choice = counts.index[0]
        dominant_percentage = percentages.iloc[0]

        return fig, dominant_choice, dominant_percentage

    def display_individual_summary(df, player_index, rules_df):
        for column in df.columns:
            if column.lower() != 'name':  # Exclude name from analysis
                col1, col2, col3, col4 = st.columns([1, 1, 1.5, 1])
                
                value = df.iloc[player_index][column]
                if isinstance(value, list):
                    value = ', '.join(map(str, value))
                elif pd.isna(value):
                    value = "N/A"
                else:
                    value = str(value)
                
                with col1:
                    st.write(f"**{column}:** {value}")
                
                # Get insight for the attribute
                insight = get_insight(column, value, rules_df)
                with col2:
                    st.info(insight)
                
                # Create comparison visualization
                with col3:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        overall_value = df[column].mean()
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=float(value),
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"{column}", 'font': {'size': 14}},
                            gauge={
                                'axis': {'range': [df[column].min(), df[column].max()], 'tickwidth': 1},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [df[column].min(), overall_value], 'color': "lightgray"},
                                    {'range': [overall_value, df[column].max()], 'color': "gray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 2},
                                    'thickness': 0.75,
                                    'value': overall_value
                                }
                            }
                        ))
                        fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                    else:
                        all_values = df[column].explode().dropna().unique()
                        overall_counts = df[column].explode().value_counts()
                        fig = go.Figure()
                        for val in all_values:
                            overall_count = overall_counts.get(val, 0)
                            player_count = 1 if val == value else 0
                            fig.add_trace(go.Bar(
                                y=[str(val)],
                                x=[overall_count],
                                orientation='h',
                                name='Overall',
                                marker_color='lightblue'
                            ))
                            fig.add_trace(go.Bar(
                                y=[str(val)],
                                x=[player_count],
                                orientation='h',
                                name='Player',
                                marker_color='darkblue'
                            ))
                        fig.update_layout(barmode='group', height=200, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display most common value
                with col4:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        overall_value = f"{df[column].mean():.2f}"
                    else:
                        overall_value = df[column].mode().iloc[0]
                        if isinstance(overall_value, list):
                            overall_value = ', '.join(map(str, overall_value))
                        elif pd.isna(overall_value):
                            overall_value = "N/A"
                        else:
                            overall_value = str(overall_value)
                    st.info(f"Most common: {overall_value}")
                
                st.markdown("---")  # Add a separator between attributes

    @st.cache_data
    def load_data():
        if not os.path.exists(csv_file):
            st.error(f"Demographics CSV file not found: {csv_file}")
            return None
        try:
            df = pd.read_csv(csv_file)
            return preprocess_data(df)
        except FileNotFoundError:
            st.error(f"File not found: {csv_file}")
            return None
        except pd.errors.EmptyDataError:
            st.error(f"The file {csv_file} is empty.")
            return None
        except Exception as e:
            st.error(f"An error occurred while reading the file: {str(e)}")
            return None

    st.title('Demographic Analysis')

    df = load_data()
    rules_df = load_insight_rules()

    if df is not None:        
        selected = option_menu(
            menu_title=None,
            options=["Overall", "Individual"],
            icons=["bar-chart-fill", "person-fill"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("# of Participants", len(df))
        with col2:
            st.metric("# of Questions", len(df.columns))
        
        if selected == "Overall":
            columns_to_plot = st.multiselect(
                'Select Question/s to Analyse',
                options=[col for col in df.columns if col.lower() != 'name'],
                default=[col for col in df.columns if col.lower() != 'name'][0]
            )
            
            if columns_to_plot:
                for column in columns_to_plot:
                    fig, dominant_choice, dominant_percentage = plot_overall_chart(df, column)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"Most common response: {dominant_choice} ({dominant_percentage:.1f}%)")
                    
                    # Get insight for the most common response
                    insight = get_insight(column, dominant_choice, rules_df)
                    st.write(f"Insight: {insight}")
            else:
                st.warning("Please select at least one question to analyse.")
        
        elif selected == "Individual":
            player_index = st.selectbox("Select Player", range(len(df)), format_func=lambda x: f"Player {x + 1}")
            display_individual_summary(df, player_index, rules_df)
            

    st.sidebar.header("About This Analysis")
    st.sidebar.info(
        "This demographic analysis tool provides insights into your participant pool for FPS gaming experiments. "
        "Use the 'Overall' view to understand general trends and the 'Individual' view to dive deep into specific participants. "
        "Insights are generated based on predefined rules, which can be customised in the insight_rules.csv file."
    )