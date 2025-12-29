"""
Advanced Travel Agent Data Preprocessing Pipeline
Handles data cleaning, feature engineering, and preparation for ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TravelDataPreprocessor:
    """Comprehensive data preprocessing for travel recommendation system"""
    
    def __init__(self, data_path='archive (7)'):
        self.data_path = data_path
        self.destinations = None
        self.reviews = None
        self.user_history = None
        self.users = None
        
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        self.destinations = pd.read_csv(f'{self.data_path}/Expanded_Destinations.csv')
        self.reviews = pd.read_csv(f'{self.data_path}/Final_Updated_Expanded_Reviews.csv')
        self.user_history = pd.read_csv(f'{self.data_path}/Final_Updated_Expanded_UserHistory.csv')
        self.users = pd.read_csv(f'{self.data_path}/Final_Updated_Expanded_Users.csv')
        print("✓ Data loaded successfully")
        
    def clean_destinations(self):
        """Clean and enhance destination data"""
        print("\nCleaning destination data...")
        
        # Handle missing values
        self.destinations['Popularity'].fillna(self.destinations['Popularity'].median(), inplace=True)
        
        # Extract season information
        season_map = {
            'Spring': 'Spring', 'Summer': 'Summer', 
            'Fall': 'Fall', 'Autumn': 'Fall',
            'Winter': 'Winter', 'Year-round': 'Year-round',
            'All': 'Year-round'
        }
        
        if 'BestTimeToVisit' in self.destinations.columns:
            self.destinations['Season'] = self.destinations['BestTimeToVisit'].map(
                lambda x: season_map.get(str(x).strip(), 'Year-round') if pd.notna(x) else 'Year-round'
            )
        
        # Categorize destinations by type
        if 'Type' in self.destinations.columns:
            self.destinations['Type'] = self.destinations['Type'].fillna('General')
        
        # Create popularity tier
        self.destinations['PopularityTier'] = pd.qcut(
            self.destinations['Popularity'], 
            q=4, 
            labels=['Hidden Gem', 'Emerging', 'Popular', 'Top Attraction'],
            duplicates='drop'
        )
        
        print(f"✓ Processed {len(self.destinations)} destinations")
        
    def aggregate_reviews(self):
        """Aggregate review statistics for each destination"""
        print("\nAggregating reviews...")
        
        # Calculate review statistics
        review_stats = self.reviews.groupby('DestinationID').agg({
            'Rating': ['mean', 'count', 'std'],
            'ReviewText': 'count'
        }).reset_index()
        
        review_stats.columns = ['DestinationID', 'AvgRating', 'ReviewCount', 'RatingStd', 'TextReviewCount']
        review_stats['RatingStd'].fillna(0, inplace=True)
        
        # Merge with destinations
        self.destinations = self.destinations.merge(review_stats, on='DestinationID', how='left')
        self.destinations['AvgRating'].fillna(3.5, inplace=True)
        self.destinations['ReviewCount'].fillna(0, inplace=True)
        
        print(f"✓ Aggregated {len(self.reviews)} reviews")
        
    def process_user_history(self):
        """Process user visit history"""
        print("\nProcessing user history...")
        
        # Convert visit date
        if 'VisitDate' in self.user_history.columns:
            self.user_history['VisitDate'] = pd.to_datetime(self.user_history['VisitDate'], errors='coerce')
            self.user_history['VisitYear'] = self.user_history['VisitDate'].dt.year
            self.user_history['VisitMonth'] = self.user_history['VisitDate'].dt.month
            self.user_history['VisitSeason'] = self.user_history['VisitMonth'].map(
                lambda x: 'Winter' if x in [12, 1, 2] else
                         'Spring' if x in [3, 4, 5] else
                         'Summer' if x in [6, 7, 8] else 'Fall'
            )
        
        # Calculate user preferences based on history
        user_dest_prefs = self.user_history.merge(
            self.destinations[['DestinationID', 'Type', 'State']], 
            on='DestinationID', 
            how='left'
        )
        
        # User's favorite types
        user_type_prefs = user_dest_prefs.groupby(['UserID', 'Type']).size().reset_index(name='VisitCount')
        self.user_favorite_types = user_type_prefs.loc[
            user_type_prefs.groupby('UserID')['VisitCount'].idxmax()
        ][['UserID', 'Type']].rename(columns={'Type': 'FavoriteType'})
        
        print(f"✓ Processed {len(self.user_history)} visit records")
        
    def enhance_user_data(self):
        """Enhance user data with derived features"""
        print("\nEnhancing user data...")
        
        # Parse preferences
        if 'Preferences' in self.users.columns:
            self.users['Preferences'] = self.users['Preferences'].fillna('General')
        
        # Calculate travel party size
        if 'NumberOfAdults' in self.users.columns and 'NumberOfChildren' in self.users.columns:
            self.users['NumberOfAdults'].fillna(1, inplace=True)
            self.users['NumberOfChildren'].fillna(0, inplace=True)
            self.users['PartySize'] = self.users['NumberOfAdults'] + self.users['NumberOfChildren']
            self.users['HasChildren'] = (self.users['NumberOfChildren'] > 0).astype(int)
        
        # Merge with favorite types
        if hasattr(self, 'user_favorite_types'):
            self.users = self.users.merge(self.user_favorite_types, on='UserID', how='left')
        
        # Calculate user activity level
        user_activity = self.user_history.groupby('UserID').size().reset_index(name='TotalVisits')
        self.users = self.users.merge(user_activity, on='UserID', how='left')
        self.users['TotalVisits'].fillna(0, inplace=True)
        
        self.users['ActivityLevel'] = pd.cut(
            self.users['TotalVisits'],
            bins=[-1, 0, 3, 7, np.inf],
            labels=['New', 'Occasional', 'Regular', 'Frequent']
        )
        
        print(f"✓ Enhanced {len(self.users)} user profiles")
        
    def create_destination_features(self):
        """Create additional features for destinations"""
        print("\nCreating destination features...")
        
        # Budget estimation based on popularity and state
        popularity_scores = self.destinations['Popularity'].values
        budget_scores = np.where(
            popularity_scores > self.destinations['Popularity'].quantile(0.75), 3,
            np.where(popularity_scores > self.destinations['Popularity'].quantile(0.5), 2, 1)
        )
        self.destinations['BudgetLevel'] = pd.Categorical(
            budget_scores, 
            categories=[1, 2, 3],
            ordered=True
        ).map({1: 'Budget', 2: 'Moderate', 3: 'Premium'})
        
        # Create composite score
        self.destinations['CompositeScore'] = (
            self.destinations['AvgRating'] * 0.4 +
            np.log1p(self.destinations['ReviewCount']) * 0.3 +
            self.destinations['Popularity'] / 100 * 0.3
        )
        
        print("✓ Created destination features")
        
    def create_interaction_matrix(self):
        """Create user-destination interaction matrix"""
        print("\nCreating interaction matrix...")
        
        # Combine explicit ratings and implicit visits
        interactions = self.user_history.merge(
            self.reviews[['UserID', 'DestinationID', 'Rating']],
            on=['UserID', 'DestinationID'],
            how='left'
        )
        
        # Use ExperienceRating if Rating is not available
        if 'ExperienceRating' in interactions.columns:
            interactions['FinalRating'] = interactions['Rating'].fillna(interactions['ExperienceRating'])
        else:
            interactions['FinalRating'] = interactions['Rating'].fillna(3.5)
        
        # Aggregate interactions - Fixed to avoid duplicate column names
        agg_dict = {
            'FinalRating': 'mean',
            'HistoryID': 'count'  # Changed from 'DestinationID' to 'HistoryID'
        }
        
        self.interaction_matrix = interactions.groupby(['UserID', 'DestinationID']).agg(agg_dict).reset_index()
        
        self.interaction_matrix.columns = ['UserID', 'DestinationID', 'AvgInteractionRating', 'InteractionCount']
        
        # Create normalized score
        self.interaction_matrix['InteractionScore'] = (
            self.interaction_matrix['AvgInteractionRating'] * 0.7 +
            np.log1p(self.interaction_matrix['InteractionCount']) * 0.3
        )
        
        print(f"✓ Created interaction matrix with {len(self.interaction_matrix)} records")
        
    def save_processed_data(self):
        """Save processed datasets"""
        print("\nSaving processed data...")
        
        self.destinations.to_csv('processed_destinations.csv', index=False)
        self.users.to_csv('processed_users.csv', index=False)
        self.interaction_matrix.to_csv('interaction_matrix.csv', index=False)
        
        # Save metadata
        metadata = {
            'destinations_count': len(self.destinations),
            'users_count': len(self.users),
            'interactions_count': len(self.interaction_matrix),
            'destination_types': self.destinations['Type'].unique().tolist() if 'Type' in self.destinations.columns else [],
            'states': self.destinations['State'].unique().tolist() if 'State' in self.destinations.columns else [],
            'processing_date': datetime.now().isoformat()
        }
        
        import json
        with open('preprocessing_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ All processed data saved successfully")
        
    def run_pipeline(self):
        """Execute complete preprocessing pipeline"""
        print("="*60)
        print("TRAVEL AGENT DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        self.load_data()
        self.clean_destinations()
        self.aggregate_reviews()
        self.process_user_history()
        self.enhance_user_data()
        self.create_destination_features()
        self.create_interaction_matrix()
        self.save_processed_data()
        
        print("\n" + "="*60)
        print("✓ PREPROCESSING COMPLETE")
        print("="*60)
        
        # Print summary
        print("\nDATA SUMMARY:")
        print(f"  • Destinations: {len(self.destinations)}")
        print(f"  • Users: {len(self.users)}")
        print(f"  • Interactions: {len(self.interaction_matrix)}")
        print(f"  • Average Rating: {self.destinations['AvgRating'].mean():.2f}")
        print(f"  • Total Reviews: {self.destinations['ReviewCount'].sum():.0f}")

if __name__ == "__main__":
    preprocessor = TravelDataPreprocessor()
    preprocessor.run_pipeline()