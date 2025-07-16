import streamlit as st
import pandas as pd
import plotly.express as px
from hn_analyzer import HNSentimentAnalyzer
import time

st.set_page_config(page_title="HN Sentiment Tracker", layout="wide")

st.title("🔥 Hacker News Sentiment Analysis")

tab1, tab2 = st.tabs(["📊 Live Analysis", "🧠 How It Works"])

with tab1:
    st.markdown("**Discover the mood of the tech community in real-time**")
    st.markdown("Analyzes sentiment of top HN headlines to show what's trending positive, negative, or neutral.")

    @st.cache_data(ttl=300)
    def load_hn_data(num_stories):
        analyzer = HNSentimentAnalyzer()
        return analyzer.analyze_stories(limit=num_stories)

    @st.cache_data(ttl=300)
    def get_summary(df):
        analyzer = HNSentimentAnalyzer()
        return analyzer.get_sentiment_summary(df)

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("**Controls**")
        num_stories = st.slider("Number of stories", 10, 50, 30)
        refresh_button = st.button("🔄 Refresh Data")

    if refresh_button:
        st.cache_data.clear()

    with st.spinner("Analyzing Hacker News stories..."):
        df = load_hn_data(num_stories)
        summary = get_summary(df)

    sentiment_map = {
        'negative': 'Negative',
        'neutral': 'Neutral', 
        'positive': 'Positive'
    }

    df['sentiment'] = df['sentiment_label'].map(sentiment_map)

    st.subheader("📈 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Stories", summary['total_stories'])

    with col2:
        positive_count = summary['sentiment_distribution'].get('positive', 0)
        st.metric("😊 Positive", positive_count, delta=f"{positive_count/summary['total_stories']*100:.0f}%")

    with col3:
        neutral_count = summary['sentiment_distribution'].get('neutral', 0)
        st.metric("😐 Neutral", neutral_count, delta=f"{neutral_count/summary['total_stories']*100:.0f}%")

    with col4:
        negative_count = summary['sentiment_distribution'].get('negative', 0)
        st.metric("😞 Negative", negative_count, delta=f"{negative_count/summary['total_stories']*100:.0f}%")

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            values=list(summary['sentiment_distribution'].values()),
            names=[sentiment_map[k] for k in summary['sentiment_distribution'].keys()],
            title="Sentiment Distribution",
            color_discrete_map={'Positive': '#2E8B57', 'Neutral': '#4682B4', 'Negative': '#CD5C5C'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.histogram(
            df, 
            x='sentiment', 
            y='score',
            title="HN Scores by Sentiment",
            color='sentiment',
            color_discrete_map={'Positive': '#2E8B57', 'Neutral': '#4682B4', 'Negative': '#CD5C5C'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("🔥 Top Stories by Sentiment")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**😊 Most Positive Stories**")
        positive_stories = df[df['sentiment'] == 'Positive'].nlargest(3, 'sentiment_score')
        if not positive_stories.empty:
            for _, story in positive_stories.iterrows():
                confidence = story['sentiment_score'] * 100
                url = story['url'] if story['url'] else f"https://news.ycombinator.com/item?id={story['id']}"
                st.markdown(f"• [{story['title'][:80]}...]({url}) - {confidence:.0f}% confident")
        else:
            st.markdown("No positive stories found in current batch")

    with col2:
        st.markdown("**😞 Most Negative Stories**")
        negative_stories = df[df['sentiment'] == 'Negative'].nlargest(3, 'sentiment_score')
        if not negative_stories.empty:
            for _, story in negative_stories.iterrows():
                confidence = story['sentiment_score'] * 100
                url = story['url'] if story['url'] else f"https://news.ycombinator.com/item?id={story['id']}"
                st.markdown(f"• [{story['title'][:80]}...]({url}) - {confidence:.0f}% confident")
        else:
            st.markdown("No negative stories found in current batch")

    st.subheader("📊 All Stories")

    sentiment_filter = st.selectbox(
        "Filter by sentiment:", 
        ["All"] + list(sentiment_map.values())
    )

    if sentiment_filter != "All":
        filtered_df = df[df['sentiment'] == sentiment_filter]
    else:
        filtered_df = df

    display_df = filtered_df.copy()
    display_df['title_link'] = display_df.apply(
        lambda row: f"[{row['title']}]({row['url'] if row['url'] else 'https://news.ycombinator.com/item?id=' + str(row['id'])})", 
        axis=1
    )
    
    st.dataframe(
        display_df[['title_link', 'sentiment', 'sentiment_score', 'score', 'by']].sort_values('sentiment_score', ascending=False),
        use_container_width=True,
        column_config={
            "title_link": st.column_config.LinkColumn("Story Title"),
            "sentiment": "Sentiment",
            "sentiment_score": st.column_config.NumberColumn("Confidence", format="%.3f"),
            "score": "HN Score",
            "by": "Author"
        }
    )

    st.markdown("---")
    st.markdown("*Data refreshes every 5 minutes. Built with Streamlit + Transformers*")

with tab2:
    st.header("🧠 How Sentiment Analysis Works")
    
    st.subheader("🎯 What We're Analyzing")
    st.markdown("""
    We fetch **top HN story headlines** and analyze their emotional tone:
    
    • **Headlines Only** - Not article content or comments  
    • **Real-time Data** - Fresh from HN Firebase API  
    • **Community Mood** - Overall sentiment in tech discussions
    """)
    
    st.subheader("📊 What Each Sentiment Means")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**😊 Positive**")
        st.markdown("""
        Optimistic, exciting language:
        • "Amazing AI breakthrough"
        • "Show HN: Revolutionary tool"
        • "Incredible improvements"
        """)
    
    with col2:
        st.markdown("**😐 Neutral**")
        st.markdown("""
        Factual, informational tone:
        • "Python 3.12 release notes"
        • "Technical memory analysis"
        • "Quarterly results announced"
        """)
    
    with col3:
        st.markdown("**😞 Negative**")
        st.markdown("""
        Concerning, critical language:
        • "Security vulnerability found"
        • "Data breach affects users"
        • "Tech layoffs continue"
        """)
    
    st.subheader("🤖 The AI Behind It")
    st.markdown("""
    **RoBERTa + Keyword Enhancement** - hybrid approach:
    
    **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
    • Pre-trained sentiment classifier
    • Enhanced with keyword detection rules
    • Aggressive anti-neutral bias corrections
    
    **Process**:
    • Run base sentiment analysis on headlines
    • Apply keyword-based overrides for obvious cases
    • Force reclassification of low-confidence neutrals
    • Boost "Show HN", launches, and tech innovations to positive
    
    **Anti-Neutral Strategy**: Combat model's conservative nature
    """)
    
    st.subheader("🔧 Technical Details")
    st.markdown("""
    • **Architecture**: RoBERTa transformer  
    • **Training**: Millions of social media posts  
    • **Output**: Probability scores (0-1)  
    • **Cache**: 5-minute refresh cycles  
    • **Source**: HN Firebase API
    """)
    
    st.subheader("⚖️ Addressing the 'Neutral Bias' Problem")
    st.markdown("""
    **Common Issue**: Most sentiment models classify everything as neutral because:
    • News headlines are often factual/objective in tone
    • Models trained on social media expect more emotional language
    • "Bitwarden is the answer" sounds neutral but is actually very positive
    
    **Our Aggressive Solutions**:
    • **Keyword Detection**: Override neutral for obvious positive/negative terms
    • **Show HN Boost**: "Show HN" posts automatically get positive sentiment
    • **Tech Launch Bias**: Product launches, releases → positive
    • **Security/Bug Bias**: Vulnerabilities, outages → negative
    • **Low Confidence Override**: Reclassify uncertain neutrals
    """)
    
    st.info("💡 Using hybrid AI + rules to force more meaningful classifications!") 