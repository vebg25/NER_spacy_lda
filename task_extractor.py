import re
import uuid
import spacy
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
import dateparser
import warnings
warnings.filterwarnings("ignore")

class TaskExtractor:
    def __init__(self):
        # Load spaCy model for NLP processing
        self.nlp = spacy.load("en_core_web_sm")

        # Task-related keywords
        self.task_verbs = ["submit", "complete", "review", "prepare", "schedule",
                           "create", "send", "organize", "clean", "update"]

        self.deadline_indicators = ["by", "before", "due", "deadline", "until", "on"]

        # Initialize date parser
        self.date_parser = dateparser

        # Word2Vec model for task embeddings
        self.word2vec = None
        self.lda_model = None
        self.vectorizer = None
        self.n_topics = 5

    def extract_tasks(self, text: str) -> List[Dict[str, Any]]:
        #Extract tasks

        # Preprocess text
        doc = self.nlp(text)

        # Extract tasks
        tasks = []
        for sent in doc.sents:
            task = self._process_sentence(sent)
            if task:
                tasks.append(task)

        # Train models if we have enough tasks
        if tasks and len(tasks) >= 2:
            self._train_categorization_models(tasks)
            tasks = self._categorize_tasks(tasks)

        return tasks

    

    def _train_categorization_models(self, tasks: List[Dict[str, Any]]):
        """Train Word2Vec and LDA models for task categorization"""
        # Prepare data for Word2Vec
        task_tokens = []
        for task in tasks:
            tokens = [token.text.lower() for token in self.nlp(task["task_description"])
                     if not token.is_stop and not token.is_punct]
            task_tokens.append(tokens)

        # Train Word2Vec model
        if len(task_tokens) >= 2:  # Need at least 2 sentences for training
            self.word2vec = Word2Vec(sentences=task_tokens, vector_size=50,
                                     window=5, min_count=1, workers=4)

        # Prepare data for LDA
        task_descriptions = [task["task_description"] for task in tasks]
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english')
        X = self.vectorizer.fit_transform(task_descriptions)

        # Determine optimal number of topics (or use fixed number for small datasets)
        n_topics = min(self.n_topics, len(tasks))

        # Train LDA model
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.lda_model.fit(X)

        # Extract topic words for naming topics
        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_names = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]

            # Name the topic based on top words
            if "report" in top_words or "document" in top_words:
                self.topic_names.append("Documentation")
            elif "meet" in top_words or "schedule" in top_words or "call" in top_words:
                self.topic_names.append("Meetings & Communication")
            elif "clean" in top_words or "organize" in top_words:
                self.topic_names.append("Maintenance & Organization")
            elif "submit" in top_words or "complete" in top_words or "assignment" in top_words:
                self.topic_names.append("Assignments & Submissions")
            elif "review" in top_words or "analyze" in top_words:
                self.topic_names.append("Review & Analysis")
            else:
                self.topic_names.append(f"Task Group {topic_idx+1}")

    def _categorize_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Categorize tasks using the trained models"""
        if not self.lda_model or not self.vectorizer:
            # Fallback categorization if models aren't trained
            for task in tasks:
                task["category"] = "General Task"
            return tasks

        # Get LDA topics for each task
        task_descriptions = [task["task_description"] for task in tasks]
        X = self.vectorizer.transform(task_descriptions)
        topic_distributions = self.lda_model.transform(X)

        # Assign topic with highest probability to each task
        for i, task in enumerate(tasks):
            topic_idx = topic_distributions[i].argmax()
            task["category"] = self.topic_names[topic_idx]

            # Add topic distribution for reference
            task["topic_distribution"] = {
                self.topic_names[j]: round(topic_distributions[i][j], 2)
                for j in range(len(self.topic_names))
            }

        return tasks