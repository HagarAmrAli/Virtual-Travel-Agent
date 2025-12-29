"""
Advanced Travel Recommendation Engine
Implements multiple recommendation algorithms with ensemble approach
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.sparse import csr_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class TravelRecommendationEngine:
    """Multi-algorithm travel recommendation system"""
    
    def __init__(self):
        self.destinations = None
        self.users = None
        self.interactions = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.svd_model = None
        self.cf_model = None
        self.content_similarity = None
        
    def load_processed_data(self):
        """Load preprocessed data"""
        print("Loading processed data...")
        self.destinations = pd.read_csv('processed_destinations.csv')
        self.users = pd.read_csv('processed_users.csv')
        self.interactions = pd.read_csv('interaction_matrix.csv')
        print("✓ Data loaded successfully")
        
    def build_content_based_features(self):
        """Build content-based feature matrix"""
        print("\nBuilding content-based features...")
        
        # Select relevant features
        feature_cols = []
        
        # Encode categorical features
        categorical_features = ['Type', 'State', 'Season', 'PopularityTier', 'BudgetLevel']
        
        for col in categorical_features:
            if col in self.destinations.columns:
                le = LabelEncoder()
                self.destinations[f'{col}_encoded'] = le.fit_transform(
                    self.destinations[col].fillna('Unknown')
                )
                self.label_encoders[col] = le
                feature_cols.append(f'{col}_encoded')
        
        # Numerical features
        numerical_features = ['Popularity', 'AvgRating', 'CompositeScore']
        for col in numerical_features:
            if col in self.destinations.columns:
                feature_cols.append(col)
        
        # Create feature matrix
        self.feature_matrix = self.destinations[feature_cols].fillna(0).values
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        # Calculate content similarity
        self.content_similarity = cosine_similarity(self.feature_matrix_scaled)
        
        print(f"✓ Created feature matrix: {self.feature_matrix_scaled.shape}")
        
    def build_collaborative_filtering(self):
        """Build collaborative filtering model using SVD"""
        print("\nBuilding collaborative filtering model...")
        
        # Create user-item matrix
        user_item = self.interactions.pivot_table(
            index='UserID',
            columns='DestinationID',
            values='InteractionScore',
            fill_value=0
        )
        
        # Convert to sparse matrix for efficiency
        sparse_user_item = csr_matrix(user_item.values)
        
        # Apply SVD for dimensionality reduction
        n_components = min(50, min(sparse_user_item.shape) - 1)
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(sparse_user_item)
        self.item_factors = self.svd_model.components_.T
        
        # Store user and item mappings
        self.user_id_map = {uid: idx for idx, uid in enumerate(user_item.index)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(user_item.columns)}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        
        print(f"✓ SVD model created with {n_components} components")
        
    def train_hybrid_model(self):
        """Train hybrid recommendation model"""
        print("\nTraining hybrid model...")
        
        # Prepare training data
        train_data = self.interactions.merge(
            self.destinations[['DestinationID', 'Popularity', 'AvgRating', 'CompositeScore']],
            on='DestinationID',
            how='left'
        )
        
        # Feature engineering for hybrid model
        features = ['Popularity', 'AvgRating', 'CompositeScore', 'InteractionCount']
        X = train_data[features].fillna(0)
        y = train_data['InteractionScore']
        
        # Train ensemble model
        self.cf_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.cf_model.fit(X, y)
        
        print("✓ Hybrid model trained successfully")
        
    def get_content_recommendations(self, destination_id, top_n=10):
        """Get content-based recommendations"""
        try:
            dest_idx = self.destinations[
                self.destinations['DestinationID'] == destination_id
            ].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity[dest_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]  # Exclude itself
            
            dest_indices = [i[0] for i in sim_scores]
            scores = [i[1] for i in sim_scores]
            
            recommendations = self.destinations.iloc[dest_indices].copy()
            recommendations['SimilarityScore'] = scores
            
            return recommendations
        except:
            return pd.DataFrame()
    
    def get_collaborative_recommendations(self, user_id, top_n=10):
        """Get collaborative filtering recommendations"""
        try:
            if user_id not in self.user_id_map:
                # Return popular items for cold start
                return self.get_popular_recommendations(top_n)
            
            user_idx = self.user_id_map[user_id]
            user_vector = self.user_factors[user_idx].reshape(1, -1)
            
            # Calculate predictions for all items
            predictions = np.dot(user_vector, self.item_factors.T).flatten()
            
            # Get user's interaction history
            user_interactions = set(
                self.interactions[self.interactions['UserID'] == user_id]['DestinationID']
            )
            
            # Get top predictions excluding already visited
            item_scores = []
            for idx, score in enumerate(predictions):
                dest_id = self.reverse_item_map.get(idx)
                if dest_id and dest_id not in user_interactions:
                    item_scores.append((dest_id, score))
            
            item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)[:top_n]
            
            recommended_ids = [x[0] for x in item_scores]
            scores = [x[1] for x in item_scores]
            
            recommendations = self.destinations[
                self.destinations['DestinationID'].isin(recommended_ids)
            ].copy()
            
            recommendations['CFScore'] = recommendations['DestinationID'].map(
                dict(zip(recommended_ids, scores))
            )
            
            return recommendations.sort_values('CFScore', ascending=False)
        except:
            return self.get_popular_recommendations(top_n)
    
    def get_popular_recommendations(self, top_n=10):
        """Get popular destinations for cold start"""
        return self.destinations.nlargest(top_n, 'CompositeScore')
    
    def recommend_by_preferences(self, preferences_dict, top_n=10):
        """
        Recommend destinations based on user preferences
        
        Parameters:
        -----------
        preferences_dict : dict
            Dictionary with keys: season, budget, interests (type), state
        top_n : int
            Number of recommendations
        """
        filtered = self.destinations.copy()
        
        # Filter by season
        if preferences_dict.get('season') and preferences_dict['season'] != 'Any':
            filtered = filtered[
                (filtered['Season'] == preferences_dict['season']) |
                (filtered['Season'] == 'Year-round')
            ]
        
        # Filter by budget
        if preferences_dict.get('budget'):
            budget_map = {
                'Budget': ['Budget'],
                'Moderate': ['Budget', 'Moderate'],
                'Premium': ['Budget', 'Moderate', 'Premium']
            }
            allowed_budgets = budget_map.get(preferences_dict['budget'], ['Budget', 'Moderate', 'Premium'])
            filtered = filtered[filtered['BudgetLevel'].isin(allowed_budgets)]
        
        # Filter by type/interests
        if preferences_dict.get('interests') and preferences_dict['interests'] != 'Any':
            filtered = filtered[filtered['Type'] == preferences_dict['interests']]
        
        # Filter by state
        if preferences_dict.get('state') and preferences_dict['state'] != 'Any':
            filtered = filtered[filtered['State'] == preferences_dict['state']]
        
        # If no results, relax constraints
        if len(filtered) == 0:
            filtered = self.destinations.copy()
        
        # Sort by composite score
        recommendations = filtered.nlargest(top_n, 'CompositeScore')
        
        # Calculate match score
        recommendations['MatchScore'] = 100
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id=None, preferences=None, top_n=10):
        """
        Get hybrid recommendations combining multiple approaches
        
        Parameters:
        -----------
        user_id : int
            User ID for personalized recommendations
        preferences : dict
            User preference dictionary
        top_n : int
            Number of recommendations
        """
        recommendations_list = []
        
        # Preference-based recommendations (highest weight)
        if preferences:
            pref_recs = self.recommend_by_preferences(preferences, top_n=top_n*2)
            pref_recs['PrefScore'] = np.linspace(1.0, 0.5, len(pref_recs))
            recommendations_list.append(pref_recs[['DestinationID', 'PrefScore']])
        
        # Collaborative filtering (if user exists)
        if user_id:
            cf_recs = self.get_collaborative_recommendations(user_id, top_n=top_n*2)
            if not cf_recs.empty and 'CFScore' in cf_recs.columns:
                cf_recs['CFScore_normalized'] = (
                    cf_recs['CFScore'] - cf_recs['CFScore'].min()
                ) / (cf_recs['CFScore'].max() - cf_recs['CFScore'].min() + 1e-10)
                recommendations_list.append(cf_recs[['DestinationID', 'CFScore_normalized']])
        
        # Combine scores
        if recommendations_list:
            combined = recommendations_list[0]
            for recs in recommendations_list[1:]:
                combined = combined.merge(recs, on='DestinationID', how='outer')
            
            # Fill NaN and calculate final score
            combined = combined.fillna(0)
            score_cols = [col for col in combined.columns if col != 'DestinationID']
            combined['FinalScore'] = combined[score_cols].sum(axis=1)
            
            # Get top recommendations
            top_dest_ids = combined.nlargest(top_n, 'FinalScore')['DestinationID']
            
            final_recs = self.destinations[
                self.destinations['DestinationID'].isin(top_dest_ids)
            ].copy()
            
            final_recs = final_recs.merge(
                combined[['DestinationID', 'FinalScore']], 
                on='DestinationID'
            )
            
            return final_recs.sort_values('FinalScore', ascending=False)
        else:
            return self.get_popular_recommendations(top_n)
    
    def save_models(self):
        """Save all trained models"""
        print("\nSaving models...")
        
        models = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'svd_model': self.svd_model,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_item_map': self.reverse_item_map,
            'cf_model': self.cf_model,
            'content_similarity': self.content_similarity,
            'feature_matrix_scaled': self.feature_matrix_scaled,
            'destinations': self.destinations,
            'users': self.users,
            'interactions': self.interactions
        }
        
        joblib.dump(models, 'travel_recommendation_model.pkl')
        print("✓ Models saved to travel_recommendation_model.pkl")
        
    def train_all_models(self):
        """Train all recommendation models"""
        print("="*60)
        print("TRAINING TRAVEL RECOMMENDATION ENGINE")
        print("="*60)
        
        self.load_processed_data()
        self.build_content_based_features()
        self.build_collaborative_filtering()
        self.train_hybrid_model()
        self.save_models()
        
        print("\n" + "="*60)
        print("✓ ALL MODELS TRAINED SUCCESSFULLY")
        print("="*60)
        
        # Test recommendations
        print("\nTesting recommendations...")
        sample_prefs = {
            'season': 'Summer',
            'budget': 'Moderate',
            'interests': 'Any',
            'state': 'Any'
        }
        test_recs = self.recommend_by_preferences(sample_prefs, top_n=5)
        print(f"\nSample recommendations: {len(test_recs)} destinations found")

if __name__ == "__main__":
    engine = TravelRecommendationEngine()
    engine.train_all_models()