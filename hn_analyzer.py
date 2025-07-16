import requests
import time
from transformers import pipeline
import pandas as pd

class HNSentimentAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.base_url = "https://hacker-news.firebaseio.com/v0"
    
    def get_top_stories(self, limit=30):
        top_stories_url = f"{self.base_url}/topstories.json"
        response = requests.get(top_stories_url)
        story_ids = response.json()[:limit]
        return story_ids
    
    def get_story_details(self, story_id):
        story_url = f"{self.base_url}/item/{story_id}.json"
        response = requests.get(story_url)
        return response.json()
    
    def analyze_stories(self, limit=30):
        story_ids = self.get_top_stories(limit)
        stories = []
        
        for story_id in story_ids:
            try:
                story = self.get_story_details(story_id)
                if story and story.get('title'):
                    title = story['title']
                    
                    sentiment = self.sentiment_analyzer(title)[0]
                    
                    label_map = {
                        'LABEL_0': 'negative',
                        'LABEL_1': 'neutral', 
                        'LABEL_2': 'positive'
                    }
                    
                    mapped_label = label_map.get(sentiment['label'], 'neutral')
                    confidence = sentiment['score']
                    
                    positive_keywords = ['show hn', 'breakthrough', 'amazing', 'incredible', 'revolutionary', 
                                       'game-changing', 'innovative', 'excellent', 'awesome', 'fantastic',
                                       'launched', 'introducing', 'new release', 'open source', 'free']
                    
                    negative_keywords = ['vulnerability', 'breach', 'hack', 'attack', 'failure', 'crash',
                                       'bug', 'problem', 'issue', 'down', 'outage', 'layoffs', 'closes',
                                       'shuts down', 'bankrupt', 'lawsuit', 'scandal']
                    
                    title_lower = title.lower()
                    
                    if any(keyword in title_lower for keyword in positive_keywords):
                        if mapped_label != 'negative' or confidence < 0.8:
                            mapped_label = 'positive'
                    elif any(keyword in title_lower for keyword in negative_keywords):
                        if mapped_label != 'positive' or confidence < 0.8:
                            mapped_label = 'negative'
                    elif mapped_label == 'neutral' and confidence < 0.7:
                        if confidence > 0.4:
                            mapped_label = 'positive' if 'LABEL_2' in sentiment['label'] else 'negative'
                    
                    stories.append({
                        'id': story_id,
                        'title': title,
                        'url': story.get('url', ''),
                        'score': story.get('score', 0),
                        'sentiment_label': mapped_label,
                        'sentiment_score': sentiment['score'],
                        'by': story.get('by', 'unknown'),
                        'time': story.get('time', 0)
                    })
                time.sleep(0.1)
            except Exception as e:
                continue
        
        return pd.DataFrame(stories)
    
    def get_sentiment_summary(self, df):
        sentiment_counts = df['sentiment_label'].value_counts()
        avg_scores = df.groupby('sentiment_label')['sentiment_score'].mean()
        
        return {
            'total_stories': len(df),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'average_sentiment_scores': avg_scores.to_dict(),
            'top_positive': df[df['sentiment_label'] == 'positive'].nlargest(3, 'sentiment_score'),
            'top_negative': df[df['sentiment_label'] == 'negative'].nlargest(3, 'sentiment_score')
        } 