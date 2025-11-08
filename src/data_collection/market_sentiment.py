"""
Market Sentiment Data Collection Module

This module collects market sentiment from:
- Financial news articles (RSS feeds)
- Reddit posts (r/wallstreetbets, r/investing, etc.)
"""

import praw
import feedparser
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentCollector:
    """
    Collects market sentiment data from news and social media
    """
    
    def __init__(
        self,
        output_dir: Path,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: Optional[str] = None
    ):
        """
        Initialize the Sentiment Collector
        
        Args:
            output_dir: Directory to save collected data
            reddit_client_id: Reddit API client ID
            reddit_client_secret: Reddit API client secret
            reddit_user_agent: Reddit API user agent
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Reddit client if credentials provided
        self.reddit = None
        if reddit_client_id and reddit_client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent or "InvestmentBot/1.0"
                )
                logger.info("Reddit client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit client: {str(e)}")
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def collect_reddit_posts(
        self,
        subreddits: List[str],
        limit: int = 100,
        time_filter: str = "month"
    ) -> pd.DataFrame:
        """
        Collect posts from Reddit subreddits
        
        Args:
            subreddits: List of subreddit names
            limit: Number of posts to collect per subreddit
            time_filter: Time filter (all, day, hour, month, week, year)
            
        Returns:
            DataFrame with Reddit post data
        """
        if not self.reddit:
            logger.warning("Reddit client not initialized. Skipping Reddit data collection.")
            return pd.DataFrame()
        
        posts_data = []
        
        logger.info(f"Collecting Reddit posts from {len(subreddits)} subreddits...")
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts
                for post in subreddit.hot(limit=limit):
                    try:
                        post_data = {
                            "source": "reddit",
                            "subreddit": subreddit_name,
                            "title": post.title,
                            "text": post.selftext,
                            "score": post.score,
                            "upvote_ratio": post.upvote_ratio,
                            "num_comments": post.num_comments,
                            "created_utc": datetime.fromtimestamp(post.created_utc),
                            "url": post.url,
                            "author": str(post.author) if post.author else "deleted",
                            "post_id": post.id,
                        }
                        posts_data.append(post_data)
                    except Exception as e:
                        logger.warning(f"Error processing post {post.id}: {str(e)}")
                        continue
                
                logger.info(f"Collected {limit} posts from r/{subreddit_name}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting data from r/{subreddit_name}: {str(e)}")
                continue
        
        if posts_data:
            df = pd.DataFrame(posts_data)
            logger.info(f"Total Reddit posts collected: {len(df)}")
            return df
        else:
            return pd.DataFrame()
    
    def collect_news_articles(
        self,
        rss_feeds: List[str]
    ) -> pd.DataFrame:
        """
        Collect financial news articles from RSS feeds
        
        Args:
            rss_feeds: List of RSS feed URLs
            
        Returns:
            DataFrame with news article data
        """
        articles_data = []
        
        logger.info(f"Collecting news articles from {len(rss_feeds)} RSS feeds...")
        
        for feed_url in rss_feeds:
            try:
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    try:
                        # Extract article text (if available)
                        article_text = ""
                        if hasattr(entry, 'summary'):
                            article_text = entry.summary
                        elif hasattr(entry, 'content'):
                            article_text = entry.content[0].value if entry.content else ""
                        
                        article_data = {
                            "source": "news",
                            "feed_url": feed_url,
                            "title": entry.title if hasattr(entry, 'title') else "",
                            "text": article_text,
                            "link": entry.link if hasattr(entry, 'link') else "",
                            "published": datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else None,
                            "author": entry.author if hasattr(entry, 'author') else "",
                        }
                        articles_data.append(article_data)
                    except Exception as e:
                        logger.warning(f"Error processing article: {str(e)}")
                        continue
                
                logger.info(f"Collected {len(feed.entries)} articles from {feed_url}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting data from {feed_url}: {str(e)}")
                continue
        
        if articles_data:
            df = pd.DataFrame(articles_data)
            logger.info(f"Total news articles collected: {len(df)}")
            return df
        else:
            return pd.DataFrame()
    
    def calculate_sentiment_scores(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Calculate sentiment scores for text data using VADER sentiment analyzer
        
        Args:
            df: DataFrame with text data
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with added sentiment scores
        """
        if df.empty:
            return df
        
        logger.info(f"Calculating sentiment scores for {len(df)} records...")
        
        sentiment_scores = []
        for text in df[text_column].fillna(""):
            scores = self.sentiment_analyzer.polarity_scores(str(text))
            sentiment_scores.append(scores)
        
        # Add sentiment columns
        df['sentiment_compound'] = [s['compound'] for s in sentiment_scores]
        df['sentiment_positive'] = [s['pos'] for s in sentiment_scores]
        df['sentiment_neutral'] = [s['neu'] for s in sentiment_scores]
        df['sentiment_negative'] = [s['neg'] for s in sentiment_scores]
        
        # Classify sentiment
        df['sentiment_label'] = df['sentiment_compound'].apply(
            lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral'
        )
        
        logger.info("Sentiment scores calculated successfully")
        return df
    
    def save_sentiment_data(self, df: pd.DataFrame, filename: str = "market_sentiment"):
        """
        Save sentiment data to CSV file
        
        Args:
            df: DataFrame with sentiment data
            filename: Output filename (without extension)
        """
        if df.empty:
            logger.warning("No sentiment data to save")
            return
        
        filepath = self.output_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved sentiment data to {filepath}")
    
    def collect_and_save(
        self,
        subreddits: List[str],
        rss_feeds: List[str],
        reddit_limit: int = 100,
        filename: str = "market_sentiment"
    ):
        """
        Collect and save sentiment data from all sources
        
        Args:
            subreddits: List of Reddit subreddits
            rss_feeds: List of RSS feed URLs
            reddit_limit: Number of posts per subreddit
            filename: Output filename
        """
        # Collect Reddit data
        reddit_df = self.collect_reddit_posts(subreddits, limit=reddit_limit)
        
        # Collect news data
        news_df = self.collect_news_articles(rss_feeds)
        
        # Combine data
        combined_df = pd.concat([reddit_df, news_df], ignore_index=True)
        
        if not combined_df.empty:
            # Calculate sentiment scores
            combined_df = self.calculate_sentiment_scores(combined_df)
            
            # Save data
            self.save_sentiment_data(combined_df, filename)
        
        return combined_df

