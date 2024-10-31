import streamlit as st
import pandas as pd
from config import LOG_FOLDER, PROCESSED_DATA_FOLDER, RAW_DATA_FOLDER
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
import pytz
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chisquare, anderson_ksamp
import gc

def show_analysis():

    # Load the datasets
    @st.cache_data
    def load_data():
        try:
            player_performance = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/player_performance.csv', parse_dates=['timestamp'])
            return player_performance
        except Exception as e:
            st.error(f"Error loading the player performance data: {str(e)}")
            return None

    @st.cache_data
    def load_mouse_data(ip_address):
        try:
            filename = f'app/import/activity_data/{ip_address}_activity_data.csv'
            return pd.read_csv(filename)
        except Exception as e:
            st.error(f"Error loading the mouse movement data for IP {ip_address}: {str(e)}")
            return None

    player_performance = load_data()

    if player_performance is not None:
        # Define timezones
        utc = pytz.UTC
        aest = pytz.timezone('Australia/Sydney')

        # Function to convert timestamp to AEST
        def to_aest(time_obj):
            if isinstance(time_obj, (int, float)):
                time_obj = datetime.fromtimestamp(time_obj)
            elif isinstance(time_obj, str):
                time_obj = pd.to_datetime(time_obj)
            
            if time_obj.tzinfo is None:
                time_obj = time_obj.replace(tzinfo=utc)
            return time_obj.astimezone(aest)

        # Convert timestamps for player performance data
        player_performance['timestamp'] = player_performance['timestamp'].apply(to_aest)

        # Function to create frequency data
        def create_frequency_data(data, columns, start_time):
            data = data.sort_values('timestamp')
            date_range = pd.date_range(start=start_time, periods=600, freq='s')
            freq_data = pd.DataFrame({'timestamp': date_range})
            
            def get_last_value(group):
                return group.iloc[-1] if len(group) > 0 else 0

            for column in columns:
                grouped = data.groupby(data['timestamp'].dt.floor('S'))[column].apply(get_last_value)
                full_range = pd.date_range(start=grouped.index.min(), end=grouped.index.max(), freq='S')
                filled_data = grouped.reindex(full_range).ffill().reset_index()
                freq_data = pd.merge(freq_data, filled_data, left_on='timestamp', right_on='index', how='left')
                freq_data[column] = freq_data[column].ffill().fillna(0)
                freq_data = freq_data.drop('index', axis=1)

            for column in columns:
                freq_data[f'{column}_diff'] = freq_data[column].diff().fillna(0)
            
            return freq_data
        
        def clean_data(df, selected_column):
            # Remove infinite values
            df = df[~np.isinf(df[f'{selected_column}_diff'])]
            
            # Convert to numeric, coercing errors to NaN
            df[f'{selected_column}_diff'] = pd.to_numeric(df[f'{selected_column}_diff'], errors='coerce')
            
            # Remove NaN values
            df = df.dropna(subset=[f'{selected_column}_diff'])
            
            return df

        def prepare_data_for_regression(df, selected_column):
            # Ensure Latency is categorical
            df['Latency'] = pd.Categorical(df['Latency'])
            
            # Create dummy variables, dropping the first to avoid perfect multicollinearity
            latency_dummies = pd.get_dummies(df['Latency'], prefix='Latency', drop_first=True)
            
            # Prepare the predictor variables
            X = sm.add_constant(latency_dummies)
            
            # Prepare the response variable (ensure it's non-negative and numeric)
            y = np.maximum(df[f'{selected_column}_diff'].astype(float), 0)
            
            # Convert X to numpy array and ensure it's float
            X = np.asarray(X).astype(float)
            
            # Ensure y is a 1D numpy array
            y = np.asarray(y).flatten()
            
            return X, y

        def run_negative_binomial(X, y):
            try:
                model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
                results = model.fit()
                return results
            except Exception as e:
                st.error(f"Error in Negative Binomial Regression: {str(e)}")
                return None

        def run_glm_poisson(X, y):
            try:
                model = sm.GLM(y, X, family=sm.families.Poisson())
                results = model.fit()
                return results
            except Exception as e:
                st.error(f"Error in GLM Poisson: {str(e)}")
                return None

        def improved_chi_square_test(df, selected_column):
            results = {}
            for latency in df['Latency'].unique():
                data = df[df['Latency'] == latency][f'{selected_column}_diff']
                
                if len(data) == 0:
                    st.warning(f"No valid data for latency {latency}")
                    continue
                
                # Use fewer bins to avoid empty bins
                observed, bins = np.histogram(data, bins='auto')
                
                # Ensure non-zero observed frequencies
                valid_indices = observed > 0
                observed = observed[valid_indices]
                bins = bins[:-1][valid_indices]
                
                if len(observed) == 0:
                    st.warning(f"No non-zero observed frequencies for latency {latency}")
                    continue
                
                lambda_mle = np.mean(data)
                expected = stats.poisson.pmf(np.arange(len(observed)), lambda_mle) * sum(observed)
                
                # Normalize expected frequencies to match the sum of observed frequencies
                expected = expected * (sum(observed) / sum(expected))
                
                try:
                    chi2, p_value = stats.chisquare(observed, expected)
                    results[latency] = {'chi2': chi2, 'p_value': p_value}
                except Exception as e:
                    st.error(f"Error in Chi-square test for latency {latency}: {str(e)}")
            
            return results

        # Streamlit app
        st.title('Comprehensive Player Performance Analysis')

        # Sidebar for selections
        st.sidebar.header('Filters')

        # Player selection
        all_players = pd.concat([player_performance['killer_ip'], player_performance['victim_ip']]).unique()
        selected_players = st.sidebar.multiselect('Select Players', all_players)

        # Latency selection
        all_latencies = player_performance['latency'].unique()
        selected_latencies = st.sidebar.multiselect('Select Latencies', all_latencies)

        # Input column selection
        input_columns = ['mouse_clicks', 'SPACE', 'A', 'W', 'S', 'D']
        selected_column = st.sidebar.selectbox('Select Input Column', input_columns)

        # Analysis selection
        st.sidebar.header('Select Analyses')
        
        st.sidebar.subheader('Recommended Methods')
        show_poisson = st.sidebar.checkbox('Poisson Regression', value=True)
        show_negative_binomial = st.sidebar.checkbox('Negative Binomial Regression')
        show_glm = st.sidebar.checkbox('Generalized Linear Model (GLM)')
        show_chi_square = st.sidebar.checkbox('Chi-square Goodness-of-fit Test')
        
        st.sidebar.subheader('Additional Methods')
        show_cdf = st.sidebar.checkbox('Show CDF')
        show_qq = st.sidebar.checkbox('Show Q-Q Plot')
        show_anova = st.sidebar.checkbox('Perform ANOVA')
        show_kruskal = st.sidebar.checkbox('Kruskal-Wallis Test')
        show_anderson = st.sidebar.checkbox('Anderson-Darling Test')
        show_bootstrap = st.sidebar.checkbox('Bootstrap Analysis')

        if selected_players and selected_latencies:
            all_freq_data = []

            for player in selected_players:
                mouse_keyboard_data = load_mouse_data(player.split('_')[1])
                if mouse_keyboard_data is None:
                    st.warning(f"No mouse movement data available for {player}")
                    continue

                mouse_keyboard_data['timestamp'] = pd.to_datetime(mouse_keyboard_data['timestamp'], unit='s').apply(to_aest)

                for latency in selected_latencies:
                    latency_data = player_performance[(player_performance['killer_ip'] == player) & 
                                                    (player_performance['latency'] == latency)]
                    if not latency_data.empty:
                        start = latency_data['timestamp'].min()
                        end = start + timedelta(minutes=10)
                        period_data = mouse_keyboard_data[(mouse_keyboard_data['timestamp'] >= start) & 
                                                        (mouse_keyboard_data['timestamp'] <= end)]
                        
                        freq_data = create_frequency_data(period_data, input_columns, start)
                        freq_data['Player'] = player
                        freq_data['Latency'] = latency
                        all_freq_data.append(freq_data)

            if all_freq_data:
                df = pd.concat(all_freq_data, ignore_index=True)

                df = clean_data(df, selected_column)

                st.subheader('Data Overview')
                st.write(f"Total number of rows: {len(df)}")
                st.write(f"Number of unique latencies: {df['Latency'].nunique()}")
                st.write(f"Data types of columns:")
                st.write(df.dtypes)
                st.write("First few rows of the data:")
                st.write(df.head())
                st.write(f"Summary statistics for {selected_column}_diff:")
                st.write(df[f'{selected_column}_diff'].describe())
                
                if show_cdf:
                    st.subheader('Cumulative Distribution Function (CDF)')
                    st.write("The CDF shows the probability that the input frequency is less than or equal to a given value. "
                            "It helps compare distributions across different latencies or players.")
                    
                    fig_cdf = go.Figure()
                    for (player, latency), group in df.groupby(['Player', 'Latency']):
                        data = group[f'{selected_column}_diff']
                        sorted_data = np.sort(data)
                        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                        fig_cdf.add_trace(go.Scatter(x=sorted_data, y=cdf, mode='lines',
                                                    name=f'{player} - {latency}ms'))
                    
                    fig_cdf.update_layout(title=f'CDF of {selected_column} (Events per Second)',
                                        xaxis_title='Events per Second',
                                        yaxis_title='Cumulative Probability')
                    st.plotly_chart(fig_cdf)
                    del fig_cdf
                    gc.collect()

                if show_qq:
                    st.subheader('Q-Q Plot')
                    st.write("The Q-Q plot compares the distribution of the data to a normal distribution. "
                            "Points following a straight line suggest the data is normally distributed.")
                    
                    fig_qq = go.Figure()
                    for (player, latency), group in df.groupby(['Player', 'Latency']):
                        data = group[f'{selected_column}_diff']
                        qq = stats.probplot(data, dist='norm')
                        fig_qq.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                                                    name=f'{player} - {latency}ms'))
                    
                    fig_qq.update_layout(title=f'Q-Q Plot of {selected_column} (Events per Second)',
                                        xaxis_title='Theoretical Quantiles',
                                        yaxis_title='Sample Quantiles')
                    st.plotly_chart(fig_qq)
                    del fig_qq
                    gc.collect()

                if show_anova:
                    st.subheader('ANOVA Results')
                    st.write("ANOVA tests if there are significant differences in the means across different groups (latencies).")
                    
                    anova_groups = [group[f'{selected_column}_diff'].values for name, group in df.groupby('Latency')]
                    f_statistic, p_value = stats.f_oneway(*anova_groups)
                    
                    st.write(f"F-statistic: {f_statistic:.4f}")
                    st.write(f"p-value: {p_value:.4f}")
                    st.write("A small p-value (< 0.05) suggests significant differences between latencies.")

                if show_poisson:
                    st.subheader('Poisson Regression Results')
                    st.write("Poisson regression models count data and can test for the effect of latency on the rate of events.")
                    
                    model = poisson(f"{selected_column}_diff ~ C(Latency)", data=df).fit()
                    st.write(model.summary())
                    
                    st.write("\nIncident Rate Ratios:")
                    st.write(np.exp(model.params))
                    st.write("\nConfidence Intervals for IRR:")
                    st.write(np.exp(model.conf_int()))

                if show_negative_binomial:
                    st.subheader('Negative Binomial Regression Results')
                    st.write("Negative Binomial regression is useful when the data is overdispersed (variance > mean).")
                    
                    X, y = prepare_data_for_regression(df, selected_column)
                    nb_results = run_negative_binomial(X, y)
                    if nb_results:
                        st.write(nb_results.summary())
                        st.write("Interpretation: Similar to Poisson regression, look at the coefficients and their p-values. "
                                "The key difference is that Negative Binomial regression allows for more variability in the data. "
                                "If the results differ significantly from the Poisson regression, it suggests that overdispersion "
                                "is present and the Negative Binomial model may be more appropriate.")

                if show_glm:
                    st.subheader('Generalized Linear Model (GLM) Results')
                    st.write("GLM with Poisson family, allowing for more complex model specifications.")
                    
                    X, y = prepare_data_for_regression(df, selected_column)
                    glm_results = run_glm_poisson(X, y)
                    if glm_results:
                        st.write(glm_results.summary())
                        st.write("Interpretation: The GLM results provide a more flexible framework for analyzing the data. "
                                "Look at the coefficients, standard errors, and p-values for each latency level. "
                                "Significant p-values (< 0.05) indicate that the corresponding latency level has a measurable effect "
                                "on the input frequency compared to the baseline. The magnitude and direction of the coefficients "
                                "show the strength and direction of these effects.")

                if show_chi_square:
                    st.subheader('Chi-square Goodness-of-fit Test Results')
                    st.write("This test compares the observed frequencies to expected frequencies under a Poisson distribution.")
                    
                    chi_square_results = improved_chi_square_test(df, selected_column)
                    if chi_square_results:
                        for latency, result in chi_square_results.items():
                            st.write(f"Latency {latency}ms - Chi-square statistic: {result['chi2']:.4f}, p-value: {result['p_value']:.4f}")
                        
                        st.write("Interpretation: For each latency level, a small p-value (< 0.05) suggests that the observed data "
                                "doesn't follow a Poisson distribution. This could indicate that the input patterns are more complex "
                                "than a simple Poisson process, possibly due to the influence of latency or other factors. "
                                "If p-values are consistently large across latencies, it suggests that a Poisson distribution "
                                "might be a good fit for modeling the input frequencies.")
                    else:
                        st.warning("No valid results from Chi-square test. This may be due to insufficient data or other issues.")

                if show_kruskal:
                    st.subheader('Kruskal-Wallis Test Results')
                    st.write("Non-parametric alternative to one-way ANOVA.")
                    
                    groups = [group[f'{selected_column}_diff'].values for name, group in df.groupby('Latency')]
                    h_statistic, p_value = stats.kruskal(*groups)
                    st.write(f"H-statistic: {h_statistic:.4f}")
                    st.write(f"p-value: {p_value:.4f}")

                if show_anderson:
                    st.subheader('Anderson-Darling Test Results')
                    st.write("This test is often more powerful than the K-S test for detecting differences in distributions.")
                    
                    groups = [group[f'{selected_column}_diff'].values for name, group in df.groupby('Latency')]
                    statistic, critical_values, significance_level = anderson_ksamp(groups)
                    st.write(f"Statistic: {statistic:.4f}")
                    st.write(f"Critical Values: {critical_values}")
                    st.write(f"Significance Level: {significance_level:.4f}")

                if show_bootstrap:
                    st.subheader('Bootstrap Analysis Results')
                    st.write("Bootstrap analysis to estimate confidence intervals for mean event rates.")
                    
                    n_iterations = 1000
                    for latency in selected_latencies:
                        data = df[df['Latency'] == latency][f'{selected_column}_diff']
                        bootstrap_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) 
                                        for _ in range(n_iterations)]
                        ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
                        st.write(f"Latency {latency}ms - Mean: {np.mean(data):.4f}, "
                                f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

            else:
                st.warning('No data available for the selected players and latencies.')

        else:
            st.warning('Please select at least one player and one latency value.')

    else:
        st.error("Cannot proceed with analysis due to data loading error.")