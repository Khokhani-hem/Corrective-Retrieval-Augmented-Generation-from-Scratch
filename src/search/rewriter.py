import re
from typing import List

class QueryRewriter:
    def rewrite(self, query: str) -> List[str]:
        cleaned_query = query.strip()
        queries = [cleaned_query]
        
        no_punct = re.sub(r"[^\w\s]", "", cleaned_query)
        if no_punct != cleaned_query and no_punct:
            queries.append(no_punct)
            
        stopwords = {
            "what", "where", "when", "which", "who", "whom", "whose",
            "why", "how", "does", "have", "with", "from", "that", "this",
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
            "am", "is", "are", "was", "were", "be", "been", "being",
            "has", "had", "having", "do", "did", "doing",
            "a", "an", "the", "and", "but", "if", "or", "because", 
            "as", "until", "while", "of", "at", "by", "for", "with",
            "about", "against", "between", "into", "through", 
            "during", "before", "after", "above", "below", "to", "from",
            "up", "down", "in", "out", "on", "off", "over", "under",
            "again", "further", "then", "once", "here", "there",
            "all", "any", "both", "each", "few", "more", "most", 
            "other", "some", "such", "no", "nor", "not", "only", 
            "own", "same", "so", "than", "too", "very",
            "can", "will", "just", "should", "now"
        }

        keywords = [
            word for word in no_punct.split()
            if len(word) > 2 and word.lower() not in stopwords
        ]

        if keywords:
            keyword_query = " ".join(keywords)
            if keyword_query not in queries:
                queries.append(keyword_query)
                
        return queries

