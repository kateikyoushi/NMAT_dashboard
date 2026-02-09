"""
NMAT Performance Analysis Dashboard
Comprehensive Policy Report Visualization System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import kruskal
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="NMAT Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING WITH CACHING
# ============================================================================

@st.cache_data(ttl=3600)
def load_data():
    """Load and prepare all datasets with caching"""
    
    # Load parquet files
    df_final = pd.read_parquet('dataset/CLEANED_FINAL_NMAT.parquet')
    df_matching = pd.read_parquet('dataset/MATCHING_PLE_NMAT.parquet')
    
    # Calculate score components
    score_components = ['NMS_VCss', 'NMS_IRss', 'NMS_Qss', 'NMS_PAss', 
                       'NMS_BIOss', 'NMS_PHYss', 'NMS_SSCss', 'NMS_CHEMss']
    
    df_final['Total_Raw_Score_Calculated'] = df_final[score_components].sum(axis=1)
    df_final['Part_I_Raw_Score'] = df_final[['NMS_VCss', 'NMS_IRss', 'NMS_Qss', 'NMS_PAss']].sum(axis=1)
    df_final['Part_II_Raw_Score'] = df_final[['NMS_BIOss', 'NMS_PHYss', 'NMS_SSCss', 'NMS_CHEMss']].sum(axis=1)
    df_final['All_Components_Present'] = df_final[score_components].notna().all(axis=1)
    
    # Integrate with PLE data
    df_matching_subset = df_matching.rename(columns={'STU_NO': 'NMA_AppNo'})
    df_final['Source_File'] = 'CLEANED_FINAL'
    
    df_integrated = df_final.merge(
        df_matching_subset[['NMA_AppNo', 'STU_RSCORE', 'STU_PRANK', 'NMAT_YEAR', 'KEY']],
        on='NMA_AppNo',
        how='left',
        indicator=True
    )
    
    df_integrated['Has_PLE_Match'] = df_integrated['_merge'] == 'both'
    
    # Filter for complete records
    df_analysis = df_integrated[df_integrated['All_Components_Present']].copy()
    
    # Convert NMS_PER to numeric
    df_analysis['NMS_PER'] = pd.to_numeric(df_analysis['NMS_PER'], errors='coerce')
    
    # Create deciles
    df_analysis['Percentile_Decile'] = pd.cut(
        df_analysis['NMS_PER'], 
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        labels=['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
        include_lowest=True
    )
    
    # Clean university type
    df_analysis['University_Type'] = df_analysis['School Type_rec2_FINAL'].fillna('Unknown')
    
    # Create course groups
    def classify_course(course):
        course_str = str(course).upper()
        if any(kw in course_str for kw in ['MEDICAL', 'ALLIED', 'NURSING', 'PHARMACY', 'HEALTH']):
            return 'Medical & Allied'
        elif any(kw in course_str for kw in ['BIOLOGY', 'NATURAL', 'SCIENCE', 'PHYSICS', 'CHEMISTRY']):
            return 'Natural Sciences'
        elif any(kw in course_str for kw in ['SOCIAL', 'BEHAVIORAL', 'PSYCHOLOGY', 'ECONOMICS']):
            return 'Social & Behavioral Sciences'
        elif any(kw in course_str for kw in ['ENGINEERING', 'TECHNOLOGY']):
            return 'Engineering & Technology'
        elif any(kw in course_str for kw in ['EDUCATION', 'TEACHER']):
            return 'Education'
        else:
            return 'Other'
    
    df_analysis['Course_Group'] = df_analysis['Course Classification'].apply(classify_course)
    
    return df_analysis, df_final, df_matching

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_metric_cards(col, label, value, delta=None):
    """Create styled metric display"""
    with col:
        st.metric(label=label, value=value, delta=delta)

def create_insight_box(text):
    """Create styled insight box"""
    st.markdown(f'<div class="insight-box">💡 <strong>Key Insight:</strong> {text}</div>', 
                unsafe_allow_html=True)

def create_warning_box(text):
    """Create styled warning box"""
    st.markdown(f'<div class="warning-box">⚠️ <strong>Note:</strong> {text}</div>', 
                unsafe_allow_html=True)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_performance_trends(df):
    """Overall Performance Trends"""
    yearly_stats = df.groupby('Year').agg({
        'Total_Raw_Score_Calculated': ['median', 'mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'Part_I_Raw_Score': ['median', 'mean'],
        'Part_II_Raw_Score': ['median', 'mean'],
        'NMS_PER': ['median', 'mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'NMA_AppNo': 'count'
    }).reset_index()
    
    yearly_stats.columns = ['Year', 'Total_Median', 'Total_Mean', 'Total_Q25', 'Total_Q75',
                            'Part1_Median', 'Part1_Mean', 'Part2_Median', 'Part2_Mean',
                            'Per_Median', 'Per_Mean', 'Per_Q25', 'Per_Q75', 'N_Examinees']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Raw Score Trends (Median with IQR)',
                       'Part I vs Part II Performance',
                       'Percentile Rank Trends',
                       'Number of Examinees per Year'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Total Raw Score
    fig.add_trace(
        go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Total_Median'],
                  mode='lines+markers', name='Median Total Score',
                  line=dict(color='steelblue', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Total_Q75'],
                  fill=None, mode='lines', line_color='lightblue',
                  showlegend=False, name='Q75'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Total_Q25'],
                  fill='tonexty', mode='lines', line_color='lightblue',
                  name='IQR', fillcolor='rgba(70, 130, 180, 0.3)'),
        row=1, col=1
    )
    
    # Plot 2: Part I vs Part II
    fig.add_trace(
        go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Part1_Median'],
                  mode='lines+markers', name='Part I (Aptitude)',
                  line=dict(color='darkgreen', width=3),
                  marker=dict(size=8, symbol='square')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Part2_Median'],
                  mode='lines+markers', name='Part II (Science)',
                  line=dict(color='darkred', width=3),
                  marker=dict(size=8, symbol='diamond')),
        row=1, col=2
    )
    
    # Plot 3: Percentile Ranks
    fig.add_trace(
        go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Per_Q75'],
                  fill=None, mode='lines', line_color='lightsalmon',
                  showlegend=False, name='Q75'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Per_Q25'],
                  fill='tonexty', mode='lines', line_color='lightsalmon',
                  name='IQR', fillcolor='rgba(255, 140, 0, 0.3)'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Per_Median'],
                  mode='lines+markers', name='Median Percentile',
                  line=dict(color='darkorange', width=3),
                  marker=dict(size=8)),
        row=2, col=1
    )
    
    # Plot 4: Examinee counts
    fig.add_trace(
        go.Bar(x=yearly_stats['Year'], y=yearly_stats['N_Examinees'],
               name='Examinees', marker_color='seagreen'),
        row=2, col=2
    )
    
    # Add trendline
    years = yearly_stats['Year'].values
    examinees = yearly_stats['N_Examinees'].values
    slope, intercept = np.polyfit(years, examinees, 1)
    trendline = slope * years + intercept
    
    fig.add_trace(
        go.Scatter(x=yearly_stats['Year'], y=trendline,
                  mode='lines', name='Trendline',
                  line=dict(color='red', width=2, dash='dash')),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    
    fig.update_yaxes(title_text="Total Raw Score", row=1, col=1)
    fig.update_yaxes(title_text="Median Raw Score", row=1, col=2)
    fig.update_yaxes(title_text="Percentile Rank", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True, hovermode='x unified')
    
    return fig, yearly_stats

def plot_stability_analysis(df):
    """Stability Analysis Boxplots"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Total Raw Scores by Year',
                       'Part I vs Part II by Year',
                       'Percentile Ranks by Year')
    )
    
    years = sorted(df['Year'].unique())
    
    # Total Raw Scores
    for year in years:
        year_data = df[df['Year'] == year]['Total_Raw_Score_Calculated'].dropna()
        fig.add_trace(
            go.Box(y=year_data, name=str(year), showlegend=False),
            row=1, col=1
        )
    
    # Part I and Part II
    for year in years:
        part1_data = df[df['Year'] == year]['Part_I_Raw_Score'].dropna()
        part2_data = df[df['Year'] == year]['Part_II_Raw_Score'].dropna()
        
        fig.add_trace(
            go.Box(y=part1_data, name=f'{year} P1', 
                  marker_color='lightgreen', showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Box(y=part2_data, name=f'{year} P2',
                  marker_color='lightcoral', showlegend=False),
            row=2, col=1
        )
    
    # Percentile Ranks
    for year in years:
        year_data = df[df['Year'] == year]['NMS_PER'].dropna()
        fig.add_trace(
            go.Box(y=year_data, name=str(year), showlegend=False),
            row=3, col=1
        )
    
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year & Test Part", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=3, col=1)
    
    fig.update_yaxes(title_text="Total Raw Score", row=1, col=1)
    fig.update_yaxes(title_text="Raw Score", row=2, col=1)
    fig.update_yaxes(title_text="Percentile Rank", row=3, col=1)
    
    fig.update_layout(height=2000, showlegend=False)
    
    return fig

def plot_decile_distribution(df):
    """Decile Distribution Analysis"""
    decile_by_year = pd.crosstab(df['Year'], df['Percentile_Decile'], normalize='index') * 100
    
    # Heatmap
    fig1 = go.Figure(data=go.Heatmap(
        z=decile_by_year.T.values,
        x=decile_by_year.index,
        y=decile_by_year.columns,
        colorscale='YlGnBu',
        text=decile_by_year.T.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Percentage (%)")
    ))
    
    fig1.update_layout(
        title="Heatmap: Year × Decile Distribution",
        xaxis_title="Year",
        yaxis_title="Percentile Decile",
        height=500
    )
    
    # Top vs Bottom Deciles Trend
    top_deciles = decile_by_year[['D8', 'D9', 'D10']].sum(axis=1)
    bottom_deciles = decile_by_year[['D1', 'D2', 'D3']].sum(axis=1)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=top_deciles.index, y=top_deciles.values,
        mode='lines+markers', name='Top Deciles (D8-D10)',
        line=dict(color='green', width=3),
        marker=dict(size=10)
    ))
    fig2.add_trace(go.Scatter(
        x=bottom_deciles.index, y=bottom_deciles.values,
        mode='lines+markers', name='Bottom Deciles (D1-D3)',
        line=dict(color='red', width=3),
        marker=dict(size=10)
    ))
    fig2.add_hline(y=30, line_dash="dash", line_color="gray",
                   annotation_text="Theoretical 30%")
    
    fig2.update_layout(
        title="Trends: Top vs Bottom Decile Representation",
        xaxis_title="Year",
        yaxis_title="Percentage (%)",
        height=500,
        hovermode='x unified'
    )
    
    return fig1, fig2, decile_by_year

def plot_university_analysis(df):
    """University Type Analysis"""
    uni_decile = pd.crosstab(df['University_Type'], df['Percentile_Decile'], 
                              normalize='index') * 100
    
    # Stacked bar chart
    fig1 = go.Figure()
    for decile in uni_decile.columns:
        fig1.add_trace(go.Bar(
            name=decile,
            y=uni_decile.index,
            x=uni_decile[decile],
            orientation='h',
            text=uni_decile[decile].round(1),
            texttemplate='%{text}%',
            textposition='inside'
        ))
    
    fig1.update_layout(
        barmode='stack',
        title="University Type → Decile Distribution (Stacked %)",
        xaxis_title="Percentage (%)",
        yaxis_title="University Type",
        height=500,
        showlegend=True
    )
    
    # Top Deciles by University Type
    top_deciles_uni = uni_decile[['D8', 'D9', 'D10']].sum(axis=1).sort_values(ascending=True)
    
    colors = ['red' if 'Foreign' in idx else 'steelblue' for idx in top_deciles_uni.index]
    
    fig2 = go.Figure(go.Bar(
        x=top_deciles_uni.values,
        y=top_deciles_uni.index,
        orientation='h',
        marker_color=colors,
        text=top_deciles_uni.values.round(2),
        texttemplate='%{text}%',
        textposition='outside'
    ))
    
    fig2.add_vline(x=30, line_dash="dash", line_color="gray",
                   annotation_text="Theoretical 30%")
    
    fig2.update_layout(
        title="Top Deciles (D8-D10) by University Type<br><sub>Foreign Examinees Highlighted in Red</sub>",
        xaxis_title="Percentage in Top Deciles (%)",
        yaxis_title="University Type",
        height=500
    )
    
    return fig1, fig2, uni_decile

def plot_course_analysis(df):
    """Course Background Analysis"""
    course_decile = pd.crosstab(df['Course_Group'], df['Percentile_Decile'],
                                 normalize='index') * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Course Classification → Decile Distribution',
                       'Top Deciles (D8-D10) by Course Background<br><sub>Medical/Science Highlighted in Green</sub>')
    )
    
    # Plot 1: Horizontal stacked bar
    deciles = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
    for i, decile in enumerate(deciles):
        fig.add_trace(
            go.Bar(
                y=course_decile.index,
                x=course_decile[decile],
                name=decile,
                orientation='h',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot 2: Top deciles comparison
    top_deciles_course = course_decile[['D8', 'D9', 'D10']].sum(axis=1).sort_values(ascending=True)
    
    colors = ['darkgreen' if 'Medical' in idx or 'Natural' in idx else 'steelblue' 
              for idx in top_deciles_course.index]
    
    fig.add_trace(
        go.Bar(
            x=top_deciles_course.values,
            y=top_deciles_course.index,
            orientation='h',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_vline(x=30, line_dash="dash", line_color="gray",
                  annotation_text="Theoretical 30%", row=2, col=1)
    
    fig.update_layout(
        height=1000,
        barmode='stack'  # Only for the first subplot
    )
    
    fig.update_xaxes(title_text="Percentage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Course Group", row=1, col=1)
    fig.update_xaxes(title_text="Percentage in Top Deciles (%)", row=2, col=1)
    fig.update_yaxes(title_text="Course Group", row=2, col=1)
    
    return fig, course_decile

def plot_ple_analysis(df):
    """PLE Integration Analysis"""
    ple_matched = df[df['Has_PLE_Match']].copy()
    non_ple = df[~df['Has_PLE_Match']].copy()
    
    # Decile distribution comparison
    all_decile = df['Percentile_Decile'].value_counts(normalize=True).sort_index() * 100
    ple_decile = ple_matched['Percentile_Decile'].value_counts(normalize=True).sort_index() * 100
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=all_decile.index,
        y=all_decile.values,
        name='All Examinees',
        marker_color='steelblue',
        opacity=0.7
    ))
    fig1.add_trace(go.Bar(
        x=ple_decile.index,
        y=ple_decile.values,
        name='PLE-Matched',
        marker_color='darkgreen',
        opacity=0.7
    ))
    fig1.add_hline(y=10, line_dash="dash", line_color="gray",
                   annotation_text="Theoretical 10%")
    
    fig1.update_layout(
        title="Decile Distribution: All vs PLE-Matched Examinees",
        xaxis_title="Percentile Decile",
        yaxis_title="Percentage (%)",
        barmode='group',
        height=500
    )
    
    # Box plot comparison
    fig2 = go.Figure()
    fig2.add_trace(go.Box(y=df['NMS_PER'], name='All', marker_color='lightblue'))
    fig2.add_trace(go.Box(y=ple_matched['NMS_PER'], name='PLE-Matched', marker_color='lightgreen'))
    if len(non_ple) > 0:
        fig2.add_trace(go.Box(y=non_ple['NMS_PER'], name='Non-Matched', marker_color='lightcoral'))
    
    fig2.update_layout(
        title="Percentile Rank Distribution by PLE Match Status",
        yaxis_title="Percentile Rank",
        height=500
    )
    
    return fig1, fig2

def plot_sankey_flow(df, source_col, target_col, title):
    """Create Sankey diagram for flow visualization"""
    
    # Prepare flow data
    flow_data = df.groupby([source_col, target_col]).size().reset_index(name='count')
    
    # Create label mappings
    sources = flow_data[source_col].unique()
    targets = flow_data[target_col].unique()
    
    all_labels = list(sources) + list(targets)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    # Map sources and targets to indices
    source_indices = [label_to_idx[s] for s in flow_data[source_col]]
    target_indices = [label_to_idx[t] for t in flow_data[target_col]]
    values = flow_data['count'].tolist()
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color='rgba(169, 169, 169, 0.2)'  # Light gray with transparency
        )
    )])
    
    fig.update_layout(
        title=title,
        height=600,
        font=dict(size=12, color='black')
    )
    
    return fig

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Load data
    with st.spinner('Loading data... Please wait.'):
        df_analysis, df_final, df_matching = load_data()
    
    # Sidebar Navigation
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.radio(
        "Select Analysis Section:",
        [
            "🏠 Executive Summary",
            "📋 Data Quality & Validation",
            "📈 Performance Trends",
            "⚖️ Stability Analysis",
            "📊 Decile Distribution",
            "🏫 University Type Analysis",
            "🎓 Course Background Analysis",
            "🏥 PLE Integration",
            "🔀 Flow Visualizations",
            "📉 Statistical Tests"
        ]
    )
    
    # Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔍 Filters")
    
    years = sorted(df_analysis['Year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Years:",
        options=years,
        default=years
    )
    
    uni_types = sorted(df_analysis['University_Type'].unique())
    selected_uni = st.sidebar.multiselect(
        "Select University Types:",
        options=uni_types,
        default=uni_types
    )
    
    # Apply filters
    if selected_years and selected_uni:
        df_filtered = df_analysis[
            (df_analysis['Year'].isin(selected_years)) &
            (df_analysis['University_Type'].isin(selected_uni))
        ]
    else:
        df_filtered = df_analysis.copy()
    
    st.sidebar.markdown(f"**Filtered Records:** {len(df_filtered):,}")
    
    # Main Content
    st.markdown('<h1 class="main-header">NMAT Performance Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("**Descriptive and Trend-Based Policy Report (2006-2018)**")
    st.markdown("---")
    
    # ========================================================================
    # PAGE: EXECUTIVE SUMMARY
    # ========================================================================
    if page == "🏠 Executive Summary":
        st.header("📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        create_metric_cards(col1, "Total Examinees", f"{len(df_filtered):,}")
        create_metric_cards(col2, "Years Covered", f"{df_filtered['Year'].nunique()}")
        create_metric_cards(col3, "Median Percentile", f"{df_filtered['NMS_PER'].median():.1f}")
        create_metric_cards(col4, "PLE Match Rate", f"{df_filtered['Has_PLE_Match'].mean()*100:.2f}%")
        
        st.markdown("---")
        
        # Key Statistics
        st.subheader("📈 Key Performance Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Score Statistics**")
            st.metric("Median Total Raw Score", f"{df_filtered['Total_Raw_Score_Calculated'].median():.0f}")
            st.metric("Median Part I Score", f"{df_filtered['Part_I_Raw_Score'].median():.0f}")
            st.metric("Median Part II Score", f"{df_filtered['Part_II_Raw_Score'].median():.0f}")
        
        with col2:
            st.markdown("**University Distribution**")
            for uni_type in ['Public', 'Private', 'Foreign']:
                count = (df_filtered['University_Type'] == uni_type).sum()
                pct = count / len(df_filtered) * 100
                st.metric(uni_type, f"{count:,}", f"{pct:.2f}%")
        
        with col3:
            st.markdown("**Course Distribution**")
            top_courses = df_filtered['Course_Group'].value_counts().head(3)
            for course, count in top_courses.items():
                pct = count / len(df_filtered) * 100
                st.metric(course, f"{count:,}", f"{pct:.1f}%")
        
        st.markdown("---")
        
        # Quick Insights
        st.subheader("💡 Key Insights")
        
        create_insight_box(
            f"The dataset contains {len(df_filtered):,} complete NMAT records spanning "
            f"{df_filtered['Year'].nunique()} years (2006-2018) with {df_filtered['Has_PLE_Match'].mean()*100:.2f}% "
            f"successfully matched to PLE data."
        )
        
        foreign_count = (df_filtered['University_Type'] == 'Foreign').sum()
        foreign_pct = foreign_count / len(df_filtered) * 100
        create_insight_box(
            f"Foreign examinees represent {foreign_pct:.2f}% of the total population "
            f"(n={foreign_count:,}), a key policy interest group."
        )
        
        medical_data = df_filtered[df_filtered['Course_Group'] == 'Medical & Allied']
        medical_median = medical_data['NMS_PER'].median()
        create_insight_box(
            f"Medical & Allied students show median percentile rank of {medical_median:.1f}, "
            f"comprising {len(medical_data)/len(df_filtered)*100:.1f}% of examinees."
        )
    
    # ========================================================================
    # PAGE: DATA QUALITY
    # ========================================================================
    elif page == "📋 Data Quality & Validation":
        st.header("📋 Data Quality & Validation")
        
        col1, col2, col3 = st.columns(3)
        
        create_metric_cards(col1, "Total Records Loaded", f"{len(df_final):,}")
        create_metric_cards(col2, "Complete Records", f"{len(df_analysis):,}")
        create_metric_cards(col3, "Completeness Rate", f"{len(df_analysis)/len(df_final)*100:.2f}%")
        
        st.markdown("---")
        
        # Component Score Completeness
        st.subheader("✅ Component Score Validation")
        
        score_components = ['NMS_VCss', 'NMS_IRss', 'NMS_Qss', 'NMS_PAss',
                           'NMS_BIOss', 'NMS_PHYss', 'NMS_SSCss', 'NMS_CHEMss']
        
        component_names = {
            'NMS_VCss': 'Verbal Comprehension',
            'NMS_IRss': 'Inductive Reasoning',
            'NMS_Qss': 'Quantitative Reasoning',
            'NMS_PAss': 'Perceptual Acuity',
            'NMS_BIOss': 'Biology',
            'NMS_PHYss': 'Physics',
            'NMS_SSCss': 'Social Science',
            'NMS_CHEMss': 'Chemistry'
        }
        
        missing_data = []
        for comp in score_components:
            missing_count = df_final[comp].isnull().sum()
            missing_pct = missing_count / len(df_final) * 100
            missing_data.append({
                'Component': component_names[comp],
                'Missing Count': missing_count,
                'Missing %': missing_pct,
                'Complete Count': len(df_final) - missing_count
            })
        
        missing_df = pd.DataFrame(missing_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Complete',
            x=missing_df['Component'],
            y=missing_df['Complete Count'],
            marker_color='lightgreen'
        ))
        fig.add_trace(go.Bar(
            name='Missing',
            x=missing_df['Component'],
            y=missing_df['Missing Count'],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            barmode='stack',
            title="Component Score Completeness",
            xaxis_title="Component",
            yaxis_title="Count",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(missing_df.style.format({
            'Missing Count': '{:,.0f}',
            'Missing %': '{:.2f}%',
            'Complete Count': '{:,.0f}'
        }), use_container_width=True)
        
        st.markdown("---")
        
        # PLE Integration Quality
        st.subheader("🔗 PLE Integration Quality")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("PLE Records Loaded", f"{len(df_matching):,}")
            st.metric("Successful Matches", f"{df_analysis['Has_PLE_Match'].sum():,}")
            st.metric("Match Rate", f"{df_analysis['Has_PLE_Match'].mean()*100:.2f}%")
        
        with col2:
            st.metric("Unmatched Records", f"{(~df_analysis['Has_PLE_Match']).sum():,}")
            st.metric("Unmatch Rate", f"{(~df_analysis['Has_PLE_Match']).mean()*100:.2f}%")
        
        create_insight_box(
            f"Data quality is excellent with {len(df_analysis)/len(df_final)*100:.2f}% "
            f"complete component scores and {df_analysis['Has_PLE_Match'].mean()*100:.2f}% "
            f"successful PLE integration."
        )
    
    # ========================================================================
    # PAGE: PERFORMANCE TRENDS
    # ========================================================================
    elif page == "📈 Performance Trends":
        st.header("📈 Overall Performance Trends (2006-2018)")
        
        fig_trends, yearly_stats = plot_performance_trends(df_filtered)
        st.plotly_chart(fig_trends, use_container_width=True)
        
        st.markdown("---")
        
        # Statistical Summary Table
        st.subheader("📊 Yearly Statistics Summary")
        
        display_stats = yearly_stats[['Year', 'Total_Median', 'Part1_Median', 
                                       'Part2_Median', 'Per_Median', 'N_Examinees']].copy()
        display_stats.columns = ['Year', 'Total Raw (Median)', 'Part I (Median)',
                                 'Part II (Median)', 'Percentile (Median)', 'N Examinees']
        
        st.dataframe(display_stats.style.format({
            'Total Raw (Median)': '{:.1f}',
            'Part I (Median)': '{:.1f}',
            'Part II (Median)': '{:.1f}',
            'Percentile (Median)': '{:.1f}',
            'N Examinees': '{:,.0f}'
        }), use_container_width=True)
        
        st.markdown("---")
        
        # Key Findings
        st.subheader("💡 Key Findings")
        
        create_insight_box(
            f"Total raw score median ranges from {yearly_stats['Total_Median'].min():.1f} "
            f"to {yearly_stats['Total_Median'].max():.1f} across the study period."
        )
        
        create_insight_box(
            f"Part I (Aptitude) scores show median range of {yearly_stats['Part1_Median'].min():.1f} "
            f"to {yearly_stats['Part1_Median'].max():.1f}, while Part II (Science) ranges from "
            f"{yearly_stats['Part2_Median'].min():.1f} to {yearly_stats['Part2_Median'].max():.1f}."
        )
        
        avg_examinees = yearly_stats['N_Examinees'].mean()
        create_insight_box(
            f"Average number of examinees per year: {avg_examinees:,.0f}, indicating "
            f"consistent examination administration throughout the period."
        )
    
    # ========================================================================
    # PAGE: STABILITY ANALYSIS
    # ========================================================================
    elif page == "⚖️ Stability Analysis":
        st.header("⚖️ Exam Score Stability Analysis")
        
        st.markdown("""
        This analysis assesses distributional stability of scores across years, 
        which serves as a proxy for exam difficulty consistency.
        """)
        
        fig_stability = plot_stability_analysis(df_filtered)
        st.plotly_chart(fig_stability, use_container_width=True)
        
        st.markdown("---")
        
        # Statistical Tests
        st.subheader("📊 Kruskal-Wallis Tests for Stability")
        
        years_list = [df_filtered[df_filtered['Year'] == y]['Total_Raw_Score_Calculated'].dropna() 
                      for y in sorted(df_filtered['Year'].unique())]
        
        if len(years_list) > 1:
            h_total, p_total = kruskal(*years_list)
            
            # Effect size (eta-squared approximation)
            n_total = len(df_filtered)
            k_groups = df_filtered['Year'].nunique()
            eta_squared = (h_total - k_groups + 1) / (n_total - k_groups) if (n_total - k_groups) > 0 else 0
            
            col1 = st.columns(1)[0]
            col2 = st.columns(1)[0]
            col3 = st.columns(1)[0]
            
            with col1:
                st.metric("H-statistic", f"{h_total:.2f}")
            with col2:
                st.metric("p-value", f"{p_total:.6f}")
            with col3:
                st.metric("Effect Size (η²)", f"{eta_squared:.4f}")
            
            if p_total < 0.05:
                create_warning_box(
                    f"Significant differences detected across years (p = {p_total:.6f}). "
                    f"This suggests distributional changes in raw scores over time."
                )
            else:
                create_insight_box(
                    f"No significant differences across years (p = {p_total:.6f}). "
                    f"Scores show distributional stability."
                )
            
            # Effect size interpretation
            if eta_squared < 0.06:
                effect_interp = "Small"
            elif eta_squared < 0.14:
                effect_interp = "Medium"
            else:
                effect_interp = "Large"
            
            create_insight_box(
                f"Effect size of η² = {eta_squared:.4f} indicates a {effect_interp.lower()} "
                f"effect magnitude of year on score distributions."
            )
        
        st.markdown("---")
        
        # Interpretation Guide
        st.subheader("📖 Interpretation Guide")
        
        st.markdown("""
        - **Boxplots** show the distribution (median, quartiles, outliers) of scores each year
        - **Median shifts** indicate central tendency changes over time
        - **IQR consistency** reflects score variability stability
        - **Kruskal-Wallis test** assesses whether year groups differ significantly
        - **Effect size (η²)** quantifies the magnitude of differences (small: <0.06, medium: 0.06-0.14, large: >0.14)
        
        **Note:** These analyses frame results as distributional stability rather than definitive difficulty changes.
        """)
    
    # ========================================================================
    # PAGE: DECILE DISTRIBUTION
    # ========================================================================
    elif page == "📊 Decile Distribution":
        st.header("📊 Decile-Based Distribution Analysis")
        
        st.markdown("""
        Decile analysis provides a policy-oriented lens for examining performance distribution patterns.
        """)
        
        fig_heat, fig_trend, decile_year = plot_decile_distribution(df_filtered)
        
        st.plotly_chart(fig_heat, use_container_width=True)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("---")
        
        # Decile Distribution Table
        st.subheader("📋 Decile Distribution by Year (%)")
        
        st.dataframe(decile_year.round(2).style.background_gradient(cmap='YlGnBu', axis=None),
                    use_container_width=True)
        
        st.markdown("---")
        
        # Top vs Bottom Deciles Summary
        st.subheader("🎯 Top vs Bottom Decile Analysis")
        
        top_deciles = decile_year[['D8', 'D9', 'D10']].sum(axis=1)
        bottom_deciles = decile_year[['D1', 'D2', 'D3']].sum(axis=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Deciles (D8-D10)**")
            st.metric("Average", f"{top_deciles.mean():.2f}%")
            st.metric("Range", f"{top_deciles.min():.2f}% - {top_deciles.max():.2f}%")
            st.metric("Std Dev", f"{top_deciles.std():.2f}%")
        
        with col2:
            st.markdown("**Bottom Deciles (D1-D3)**")
            st.metric("Average", f"{bottom_deciles.mean():.2f}%")
            st.metric("Range", f"{bottom_deciles.min():.2f}% - {bottom_deciles.max():.2f}%")
            st.metric("Std Dev", f"{bottom_deciles.std():.2f}%")
        
        create_insight_box(
            f"Top deciles average {top_deciles.mean():.2f}% (theoretical: 30%), while "
            f"bottom deciles average {bottom_deciles.mean():.2f}% (theoretical: 30%). "
            f"Deviations from theoretical values indicate distribution shifts."
        )
    
    # ========================================================================
    # PAGE: UNIVERSITY ANALYSIS
    # ========================================================================
    elif page == "🏫 University Type Analysis":
        st.header("🏫 University Type → Decile Distribution")
        
        st.markdown("""
        Analysis of performance patterns across university types with **special emphasis on Foreign examinees**.
        """)
        
        fig_uni_stack, fig_uni_top, uni_decile = plot_university_analysis(df_filtered)
        
        st.plotly_chart(fig_uni_stack, use_container_width=True)
        st.plotly_chart(fig_uni_top, use_container_width=True)
        
        st.markdown("---")
        
        # Foreign Examinees Deep Dive
        st.subheader("🌍 Foreign Examinees Analysis (Policy Interest)")
        
        foreign_data = df_filtered[df_filtered['University_Type'] == 'Foreign']
        
        if len(foreign_data) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Foreign Examinees", f"{len(foreign_data):,}")
            with col2:
                st.metric("% of Total", f"{len(foreign_data)/len(df_filtered)*100:.2f}%")
            with col3:
                st.metric("Median Percentile", f"{foreign_data['NMS_PER'].median():.1f}")
            with col4:
                top_decile_foreign = foreign_data['Percentile_Decile'].isin(['D8', 'D9', 'D10']).mean() * 100
                st.metric("Top Decile %", f"{top_decile_foreign:.2f}%")
            
            # Foreign decile distribution
            foreign_decile = foreign_data['Percentile_Decile'].value_counts(normalize=True).sort_index() * 100
            
            fig_foreign = go.Figure(go.Bar(
                x=foreign_decile.index,
                y=foreign_decile.values,
                marker_color='red',
                text=foreign_decile.values.round(1),
                texttemplate='%{text}%',
                textposition='outside'
            ))
            fig_foreign.add_hline(y=10, line_dash="dash", line_color="gray",
                                  annotation_text="Theoretical 10%")
            fig_foreign.update_layout(
                title="Foreign Examinees: Decile Distribution",
                xaxis_title="Decile",
                yaxis_title="Percentage (%)",
                height=400
            )
            
            st.plotly_chart(fig_foreign, use_container_width=True)
            
            create_insight_box(
                f"Foreign examinees (n={len(foreign_data):,}) represent {len(foreign_data)/len(df_filtered)*100:.2f}% "
                f"of the population with median percentile rank of {foreign_data['NMS_PER'].median():.1f}. "
                f"Their top decile representation is {top_decile_foreign:.2f}%."
            )
        else:
            create_warning_box("No Foreign examinees found in the filtered dataset.")
        
        st.markdown("---")
        
        # University Type Comparison Table
        st.subheader("📊 University Type Performance Summary")
        
        uni_summary = []
        for uni_type in df_filtered['University_Type'].unique():
            uni_data = df_filtered[df_filtered['University_Type'] == uni_type]
            uni_summary.append({
                'University Type': uni_type,
                'Count': len(uni_data),
                'Percentage': len(uni_data) / len(df_filtered) * 100,
                'Median Percentile': uni_data['NMS_PER'].median(),
                'Mean GPS': uni_data['NMS_GPS'].mean(),
                'Top Decile %': uni_data['Percentile_Decile'].isin(['D8', 'D9', 'D10']).mean() * 100
            })
        
        uni_summary_df = pd.DataFrame(uni_summary).sort_values('Count', ascending=False)
        
        st.dataframe(uni_summary_df.style.format({
            'Count': '{:,.0f}',
            'Percentage': '{:.2f}%',
            'Median Percentile': '{:.1f}',
            'Mean GPS': '{:.1f}',
            'Top Decile %': '{:.2f}%'
        }).background_gradient(subset=['Median Percentile'], cmap='RdYlGn'),
                    use_container_width=True)
    
    # ========================================================================
    # PAGE: COURSE ANALYSIS
    # ========================================================================
    elif page == "🎓 Course Background Analysis":
        st.header("🎓 Pre-Med/Course Background → Decile Distribution")
        
        fig_course, course_decile = plot_course_analysis(df_filtered)
        st.plotly_chart(fig_course, use_container_width=True)
        
        st.markdown("---")
        
        # Course Group Comparison
        st.subheader("📊 Course Group Performance Summary")
        
        course_summary = []
        for course_group in df_filtered['Course_Group'].unique():
            course_data = df_filtered[df_filtered['Course_Group'] == course_group]
            course_summary.append({
                'Course Group': course_group,
                'Count': len(course_data),
                'Percentage': len(course_data) / len(df_filtered) * 100,
                'Median Percentile': course_data['NMS_PER'].median(),
                'Mean GPS': course_data['NMS_GPS'].mean(),
                'Top Decile %': course_data['Percentile_Decile'].isin(['D8', 'D9', 'D10']).mean() * 100,
                'Bottom Decile %': course_data['Percentile_Decile'].isin(['D1', 'D2', 'D3']).mean() * 100
            })
        
        course_summary_df = pd.DataFrame(course_summary).sort_values('Median Percentile', ascending=False)
        
        st.dataframe(course_summary_df.style.format({
            'Count': '{:,.0f}',
            'Percentage': '{:.2f}%',
            'Median Percentile': '{:.1f}',
            'Mean GPS': '{:.1f}',
            'Top Decile %': '{:.2f}%',
            'Bottom Decile %': '{:.2f}%'
        }).background_gradient(subset=['Median Percentile', 'Top Decile %'], cmap='RdYlGn'),
                    use_container_width=True)
        
        st.markdown("---")
        
        # Medical & Allied vs Others
        st.subheader("🏥 Medical & Allied vs Other Courses")
        
        medical_data = df_filtered[df_filtered['Course_Group'] == 'Medical & Allied']
        other_data = df_filtered[df_filtered['Course_Group'] != 'Medical & Allied']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Medical & Allied**")
            st.metric("Count", f"{len(medical_data):,}")
            st.metric("Median Percentile", f"{medical_data['NMS_PER'].median():.1f}")
            st.metric("Top Decile %", 
                     f"{medical_data['Percentile_Decile'].isin(['D8', 'D9', 'D10']).mean()*100:.2f}%")
        
        with col2:
            st.markdown("**Other Courses**")
            st.metric("Count", f"{len(other_data):,}")
            st.metric("Median Percentile", f"{other_data['NMS_PER'].median():.1f}")
            st.metric("Top Decile %",
                     f"{other_data['Percentile_Decile'].isin(['D8', 'D9', 'D10']).mean()*100:.2f}%")
        
        # Statistical test
        if len(medical_data) > 0 and len(other_data) > 0:
            u_stat, p_val = stats.mannwhitneyu(medical_data['NMS_PER'].dropna(),
                                                other_data['NMS_PER'].dropna(),
                                                alternative='two-sided')
            
            if p_val < 0.05:
                create_insight_box(
                    f"Mann-Whitney U test shows significant difference between Medical & Allied "
                    f"and other courses (U={u_stat:.0f}, p={p_val:.6f})."
                )
            else:
                create_warning_box(
                    f"No significant difference detected between course groups "
                    f"(U={u_stat:.0f}, p={p_val:.6f})."
                )
    
    # ========================================================================
    # PAGE: PLE INTEGRATION
    # ========================================================================
    elif page == "🏥 PLE Integration":
        st.header("🏥 PLE Integration Analysis")
        
        st.markdown("""
        Descriptive alignment with PLE (Physician Licensure Examination) passers using existing matches.
        """)
        
        ple_matched = df_filtered[df_filtered['Has_PLE_Match']].copy()
        non_ple = df_filtered[~df_filtered['Has_PLE_Match']].copy()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(df_filtered):,}")
        with col2:
            st.metric("PLE Matched", f"{len(ple_matched):,}")
        with col3:
            st.metric("Match Rate", f"{len(ple_matched)/len(df_filtered)*100:.2f}%")
        
        st.markdown("---")
        
        if len(ple_matched) > 0:
            fig_ple1, fig_ple2 = plot_ple_analysis(df_filtered)
            
            st.plotly_chart(fig_ple1, use_container_width=True)
            st.plotly_chart(fig_ple2, use_container_width=True)
            
            st.markdown("---")
            
            # Performance Comparison
            st.subheader("📊 Performance Comparison: PLE-Matched vs Non-Matched")
            
            comparison_data = {
                'Metric': [
                    'Median Percentile',
                    'Mean GPS',
                    'Mean Total Raw Score',
                    'Top Decile %'
                ],
                'PLE-Matched': [
                    ple_matched['NMS_PER'].median(),
                    ple_matched['NMS_GPS'].mean(),
                    ple_matched['Total_Raw_Score_Calculated'].mean(),
                    ple_matched['Percentile_Decile'].isin(['D8', 'D9', 'D10']).mean() * 100
                ]
            }
            
            if len(non_ple) > 0:
                comparison_data['Non-Matched'] = [
                    non_ple['NMS_PER'].median(),
                    non_ple['NMS_GPS'].mean(),
                    non_ple['Total_Raw_Score_Calculated'].mean(),
                    non_ple['Percentile_Decile'].isin(['D8', 'D9', 'D10']).mean() * 100
                ]
            
            comparison_df = pd.DataFrame(comparison_data)
            
            st.dataframe(comparison_df.style.format({
                'PLE-Matched': '{:.2f}',
                'Non-Matched': '{:.2f}' if 'Non-Matched' in comparison_df.columns else None
            }).background_gradient(subset=['PLE-Matched'], cmap='Greens'),
                        use_container_width=True)
            
            # Statistical test
            if len(non_ple) > 0:
                u_stat, p_val = stats.mannwhitneyu(ple_matched['NMS_PER'].dropna(),
                                                    non_ple['NMS_PER'].dropna(),
                                                    alternative='two-sided')
                
                st.markdown("---")
                st.subheader("📈 Statistical Test: Mann-Whitney U")
                
                col1 = st.columns(1)[0]
                col2 = st.columns(1)[0]
                with col1:
                    st.metric("U-statistic", f"{u_stat:.0f}")
                with col2:
                    st.metric("p-value", f"{p_val:.6f}")
                
                if p_val < 0.05:
                    create_insight_box(
                        f"PLE-matched examinees show significantly different performance "
                        f"compared to non-matched (p={p_val:.6f})."
                    )
            
            create_insight_box(
                f"High PLE match rate ({len(ple_matched)/len(df_filtered)*100:.2f}%) indicates "
                f"excellent data integration quality and comprehensive coverage of the NMAT-PLE pathway."
            )
        else:
            create_warning_box("No PLE-matched records found in filtered dataset.")
    
    # ========================================================================
    # PAGE: FLOW VISUALIZATIONS
    # ========================================================================
    elif page == "🔀 Flow Visualizations":
        st.header("🔀 Alluvial/Flow Visualizations")
        
        st.markdown("""
        Flow diagrams illustrate performance pathways and "survival" into higher deciles.
        These visualizations show aggregate patterns, not individual trajectories.
        """)
        
        # Tabs for different flows
        tab1, tab2, tab3 = st.tabs([
            "🏫 University Type → Decile",
            "🎓 Course Group → Decile",
            "🏥 Decile → PLE Status"
        ])
        
        with tab1:
            st.subheader("University Type → Decile Distribution Flow")
            
            fig_sankey_uni = plot_sankey_flow(
                df_filtered,
                'University_Type',
                'Percentile_Decile',
                'Flow: University Type → Percentile Decile'
            )
            st.plotly_chart(fig_sankey_uni, use_container_width=True)
            
            # Top pathways
            top_flows_uni = df_filtered.groupby(['University_Type', 'Percentile_Decile']).size().reset_index(name='count')
            top_flows_uni = top_flows_uni[top_flows_uni['Percentile_Decile'].isin(['D8', 'D9', 'D10'])]
            top_flows_uni = top_flows_uni.sort_values('count', ascending=False).head(10)
            
            st.markdown("**Top 10 Pathways to Top Deciles (D8-D10)**")
            st.dataframe(top_flows_uni.style.format({'count': '{:,.0f}'}),
                        use_container_width=True)
        
        with tab2:
            st.subheader("Course Group → Decile Distribution Flow")
            
            fig_sankey_course = plot_sankey_flow(
                df_filtered,
                'Course_Group',
                'Percentile_Decile',
                'Flow: Course Group → Percentile Decile'
            )
            st.plotly_chart(fig_sankey_course, use_container_width=True)
            
            # Survival rates
            survival_data = []
            for course in df_filtered['Course_Group'].unique():
                course_data = df_filtered[df_filtered['Course_Group'] == course]
                total = len(course_data)
                top_decile = course_data['Percentile_Decile'].isin(['D8', 'D9', 'D10']).sum()
                survival_rate = (top_decile / total * 100) if total > 0 else 0
                survival_data.append({
                    'Course Group': course,
                    'Total': total,
                    'Top Decile Count': top_decile,
                    'Survival Rate (%)': survival_rate
                })
            
            survival_df = pd.DataFrame(survival_data).sort_values('Survival Rate (%)', ascending=False)
            
            st.markdown("**Survival Rate to Top Deciles by Course Group**")
            st.dataframe(survival_df.style.format({
                'Total': '{:,.0f}',
                'Top Decile Count': '{:,.0f}',
                'Survival Rate (%)': '{:.2f}%'
            }).background_gradient(subset=['Survival Rate (%)'], cmap='RdYlGn'),
                        use_container_width=True)
        
        with tab3:
            st.subheader("Decile → PLE Match Status Flow")
            
            # Create PLE status flow
            decile_ple_flow = df_filtered.groupby(['Percentile_Decile', 'Has_PLE_Match']).size().reset_index(name='count')
            decile_ple_flow['PLE_Status'] = decile_ple_flow['Has_PLE_Match'].map({
                True: 'PLE Matched',
                False: 'Not Matched'
            })
            
            fig_sankey_ple = plot_sankey_flow(
                decile_ple_flow,
                'Percentile_Decile',
                'PLE_Status',
                'Flow: Percentile Decile → PLE Match Status'
            )
            
            # Replace PLE Status with proper labels
            fig_sankey_ple.data[0].node.label = [
                label if not isinstance(label, bool) else ('PLE Matched' if label else 'Not Matched')
                for label in fig_sankey_ple.data[0].node.label
            ]
            
            st.plotly_chart(fig_sankey_ple, use_container_width=True)
            
            # Non-match rates by decile
            non_match_data = []
            for decile in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']:
                decile_data = df_filtered[df_filtered['Percentile_Decile'] == decile]
                if len(decile_data) > 0:
                    non_match_count = (~decile_data['Has_PLE_Match']).sum()
                    non_match_rate = (non_match_count / len(decile_data)) * 100
                    non_match_data.append({
                        'Decile': decile,
                        'Total': len(decile_data),
                        'Not Matched': non_match_count,
                        'Non-Match Rate (%)': non_match_rate
                    })
            
            non_match_df = pd.DataFrame(non_match_data)
            
            st.markdown("**PLE Non-Match Rates by Decile**")
            st.dataframe(non_match_df.style.format({
                'Total': '{:,.0f}',
                'Not Matched': '{:,.0f}',
                'Non-Match Rate (%)': '{:.3f}%'
            }).background_gradient(subset=['Non-Match Rate (%)'], cmap='RdYlGn_r'),
                        use_container_width=True)
            
            create_insight_box(
                f"Near-universal PLE matching ({df_filtered['Has_PLE_Match'].mean()*100:.2f}%) "
                f"indicates strong data integration across performance levels."
            )
    
    # ========================================================================
    # PAGE: STATISTICAL TESTS
    # ========================================================================
    elif page == "📉 Statistical Tests":
        st.header("📉 Statistical Tests Summary")
        
        st.markdown("""
        Comprehensive statistical testing using non-parametric methods treating 
        percentile ranks and deciles as ordinal variables.
        """)
        
        # Kruskal-Wallis by Year
        st.subheader("🔬 Kruskal-Wallis Test: Percentile Ranks by Year")
        
        years_list = [df_filtered[df_filtered['Year'] == y]['NMS_PER'].dropna()
                     for y in sorted(df_filtered['Year'].unique())]
        
        if len(years_list) > 1:
            h_stat, p_val = kruskal(*years_list)
            
            n_total = len(df_filtered)
            k_groups = df_filtered['Year'].nunique()
            eta_squared = (h_stat - k_groups + 1) / (n_total - k_groups) if (n_total - k_groups) > 0 else 0
            
            col1 = st.columns(1)[0]
            col2 = st.columns(1)[0]
            col3 = st.columns(1)[0]
            col4 = st.columns(1)[0]
            
            with col1:
                st.metric("H-statistic", f"{h_stat:.2f}")
            with col2:
                st.metric("p-value", f"{p_val:.6f}")
            with col3:
                st.metric("df", f"{k_groups - 1}")
            with col4:
                st.metric("Effect Size (η²)", f"{eta_squared:.4f}")
            
            if p_val < 0.05:
                create_warning_box(
                    f"Significant differences detected across years (p < 0.05). "
                    f"This indicates distributional shifts in performance over time."
                )
            else:
                create_insight_box(
                    f"No significant differences across years (p ≥ 0.05). "
                    f"This suggests distributional stability."
                )
        
        st.markdown("---")
        
        # Kruskal-Wallis by University Type
        st.subheader("🔬 Kruskal-Wallis Test: Percentile Ranks by University Type")
        
        uni_list = [df_filtered[df_filtered['University_Type'] == u]['NMS_PER'].dropna()
                   for u in df_filtered['University_Type'].unique()]
        
        if len(uni_list) > 1:
            h_uni, p_uni = kruskal(*uni_list)
            
            col1 = st.columns(1)[0]
            col2 = st.columns(1)[0]
            col3 = st.columns(1)[0]
            
            with col1:
                st.metric("H-statistic", f"{h_uni:.2f}")
            with col2:
                st.metric("p-value", f"{p_uni:.6f}")
            with col3:
                st.metric("df", f"{len(uni_list) - 1}")
            
            if p_uni < 0.05:
                create_insight_box(
                    f"Significant differences detected across university types (p < 0.05)."
                )
            else:
                create_warning_box(
                    f"No significant differences across university types (p ≥ 0.05)."
                )
        
        st.markdown("---")
        
        # Chi-square Test: University × Decile
        st.subheader("🔬 Chi-Square Test: University Type × Decile Independence")
        
        contingency_table = pd.crosstab(df_filtered['University_Type'], 
                                        df_filtered['Percentile_Decile'])
        
        chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("χ² statistic", f"{chi2:.2f}")
        with col2:
            st.metric("p-value", f"{p_chi:.6f}")
        with col3:
            st.metric("df", f"{dof}")
        
        if p_chi < 0.05:
            create_insight_box(
                f"Significant association between University Type and Decile distribution (p < 0.05). "
                f"Performance patterns vary by institution type."
            )
        
        st.markdown("---")
        
        # Interpretation Guide
        st.subheader("📖 Statistical Test Interpretation")
        
        st.markdown("""
        **Kruskal-Wallis Test**
        - Non-parametric alternative to one-way ANOVA
        - Tests whether groups come from the same distribution
        - Does not assume normality
        - p < 0.05: Significant differences exist between groups
        
        **Effect Size (η²)**
        - Small: < 0.06
        - Medium: 0.06 - 0.14
        - Large: > 0.14
        
        **Chi-Square Test**
        - Tests independence between two categorical variables
        - p < 0.05: Variables are associated (not independent)
        - Used for contingency table analysis
        
        **Mann-Whitney U Test**
        - Non-parametric alternative to independent t-test
        - Compares two independent groups
        - Tests whether one group tends to have larger values
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p><strong>NMAT Performance Analysis Dashboard</strong></p>
        <p>Data Period: 2006-2018 | Records: {:,} | Years: {}</p>
        <p>Developed for Policy Report and Stakeholder Analysis</p>
    </div>
    """.format(len(df_analysis), df_analysis['Year'].nunique()), unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()