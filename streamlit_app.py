"""
Advanced Virtual Travel Agent - Streamlit Interface
Professional travel recommendation system with interactive UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Virtual Travel Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .destination-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .recommendation-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
        transition: all 0.3s;
    }
    .recommendation-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-color: #1E88E5;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

class TravelAgentApp:
    """Main application class"""
    
    def __init__(self):
        self.load_models()
        self.initialize_session_state()
    
    def load_models(self):
        """Load trained models and data"""
        try:
            models = joblib.load('travel_recommendation_model.pkl')
            self.destinations = models['destinations']
            self.users = models['users']
            self.models = models
            st.success("✓ Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.stop()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = None
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">✈️ Virtual Travel Agent</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">Discover your perfect destination with AI-powered recommendations</p>',
            unsafe_allow_html=True
        )
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with user inputs"""
        with st.sidebar:
            st.header("🎯 Your Travel Preferences")
            
            # User selection
            user_mode = st.radio(
                "Travel Mode:",
                ["New Traveler", "Returning User"],
                help="Select if you're a new or returning user"
            )
            
            if user_mode == "Returning User":
                user_ids = sorted(self.users['UserID'].unique())
                selected_user = st.selectbox(
                    "Select Your Profile:",
                    options=user_ids,
                    format_func=lambda x: f"User {x}"
                )
                st.session_state.user_id = selected_user
                
                # Display user info
                user_info = self.users[self.users['UserID'] == selected_user].iloc[0]
                st.info(f"""
                **Profile Info:**
                - Name: {user_info.get('Name', 'N/A')}
                - Party Size: {user_info.get('PartySize', 'N/A')}
                - Activity Level: {user_info.get('ActivityLevel', 'N/A')}
                """)
            else:
                st.session_state.user_id = None
            
            st.markdown("---")
            
            # Travel preferences
            st.subheader("📅 When & Where")
            
            season = st.selectbox(
                "Preferred Season:",
                ["Any", "Spring", "Summer", "Fall", "Winter", "Year-round"],
                help="Best time to visit"
            )
            
            # Get unique states
            states = ["Any"] + sorted(self.destinations['State'].dropna().unique().tolist())
            state = st.selectbox(
                "Preferred State/Region:",
                states,
                help="Choose a specific state or region"
            )
            
            st.markdown("---")
            st.subheader("💰 Budget & Interests")
            
            budget = st.select_slider(
                "Budget Level:",
                options=["Budget", "Moderate", "Premium"],
                value="Moderate",
                help="Your travel budget range"
            )
            
            # Get unique types
            types = ["Any"] + sorted(self.destinations['Type'].dropna().unique().tolist())
            interests = st.selectbox(
                "Travel Interests:",
                types,
                help="Type of destination you're interested in"
            )
            
            party_size = st.number_input(
                "Party Size:",
                min_value=1,
                max_value=20,
                value=2,
                help="Total number of travelers"
            )
            
            has_children = st.checkbox(
                "Traveling with Children",
                help="Check if traveling with kids"
            )
            
            st.markdown("---")
            
            num_recommendations = st.slider(
                "Number of Recommendations:",
                min_value=5,
                max_value=20,
                value=10,
                help="How many destinations to recommend"
            )
            
            preferences = {
                'season': season,
                'state': state,
                'budget': budget,
                'interests': interests,
                'party_size': party_size,
                'has_children': has_children
            }
            
            # Get recommendations button
            if st.button("🔍 Find Destinations", type="primary"):
                with st.spinner("Finding your perfect destinations..."):
                    recommendations = self.get_recommendations(
                        preferences, 
                        num_recommendations
                    )
                    st.session_state.recommendations = recommendations
                    st.success(f"Found {len(recommendations)} amazing destinations!")
            
            return preferences
    
    def get_recommendations(self, preferences, top_n=10):
        """Get recommendations based on preferences"""
        from travel_recommendation_engine import TravelRecommendationEngine
        
        engine = TravelRecommendationEngine()
        engine.destinations = self.destinations
        engine.users = self.users
        engine.interactions = self.models['interactions']
        engine.scaler = self.models['scaler']
        engine.label_encoders = self.models['label_encoders']
        engine.svd_model = self.models['svd_model']
        engine.user_factors = self.models['user_factors']
        engine.item_factors = self.models['item_factors']
        engine.user_id_map = self.models['user_id_map']
        engine.item_id_map = self.models['item_id_map']
        engine.reverse_item_map = self.models['reverse_item_map']
        engine.content_similarity = self.models['content_similarity']
        
        # Get hybrid recommendations
        recommendations = engine.get_hybrid_recommendations(
            user_id=st.session_state.user_id,
            preferences=preferences,
            top_n=top_n
        )
        
        return recommendations
    
    def render_statistics(self):
        """Render statistics dashboard"""
        st.header("📊 Travel Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Destinations",
                len(self.destinations),
                help="Available destinations in database"
            )
        
        with col2:
            avg_rating = self.destinations['AvgRating'].mean()
            st.metric(
                "Average Rating",
                f"{avg_rating:.2f}⭐",
                help="Average destination rating"
            )
        
        with col3:
            total_reviews = self.destinations['ReviewCount'].sum()
            st.metric(
                "Total Reviews",
                f"{total_reviews:,.0f}",
                help="Total user reviews"
            )
        
        with col4:
            popular = len(self.destinations[self.destinations['Popularity'] > 75])
            st.metric(
                "Top Attractions",
                popular,
                help="High popularity destinations"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Destinations by type
            type_counts = self.destinations['Type'].value_counts().head(10)
            fig = px.bar(
                x=type_counts.values,
                y=type_counts.index,
                orientation='h',
                title="Top 10 Destination Types",
                labels={'x': 'Count', 'y': 'Type'},
                color=type_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rating distribution
            fig = px.histogram(
                self.destinations,
                x='AvgRating',
                nbins=20,
                title="Rating Distribution",
                labels={'AvgRating': 'Average Rating', 'count': 'Frequency'},
                color_discrete_sequence=['#1E88E5']
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_recommendations(self):
        """Render recommendation results"""
        if st.session_state.recommendations is None:
            st.info("👈 Use the sidebar to set your preferences and get personalized recommendations!")
            return
        
        recommendations = st.session_state.recommendations
        
        st.header(f"🎯 Your Top {len(recommendations)} Recommendations")
        st.markdown("---")
        
        # Display recommendations
        for idx, row in recommendations.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.subheader(f"🏖️ {row['Name']}")
                    st.write(f"**Location:** {row.get('State', 'N/A')}")
                    st.write(f"**Type:** {row.get('Type', 'N/A')}")
                    st.write(f"**Best Season:** {row.get('Season', 'N/A')}")
                
                with col2:
                    st.metric("Rating", f"{row['AvgRating']:.1f}⭐")
                    st.metric("Popularity", f"{row['Popularity']:.0f}")
                
                with col3:
                    st.metric("Budget", row.get('BudgetLevel', 'N/A'))
                    st.metric("Reviews", f"{row['ReviewCount']:.0f}")
                
                # Feedback section
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button("👍 Interested", key=f"like_{row['DestinationID']}"):
                        self.record_feedback(row['DestinationID'], 'interested')
                        st.success("Added to your interested list!")
                
                with col2:
                    if st.button("👎 Not for me", key=f"dislike_{row['DestinationID']}"):
                        self.record_feedback(row['DestinationID'], 'not_interested')
                        st.info("We'll adjust future recommendations")
                
                st.markdown("---")
    
    def record_feedback(self, destination_id, feedback_type):
        """Record user feedback"""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'user_id': st.session_state.user_id,
            'destination_id': destination_id,
            'feedback': feedback_type
        }
        st.session_state.feedback_history.append(feedback)
        
        # Save feedback to file
        try:
            with open('user_feedback.json', 'a') as f:
                json.dump(feedback, f)
                f.write('\n')
        except:
            pass
    
    def render_explore_tab(self):
        """Render explore destinations tab"""
        st.header("🗺️ Explore All Destinations")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            types = ["All"] + sorted(self.destinations['Type'].dropna().unique().tolist())
            selected_type = st.selectbox("Filter by Type:", types)
        
        with col2:
            states = ["All"] + sorted(self.destinations['State'].dropna().unique().tolist())
            selected_state = st.selectbox("Filter by State:", states)
        
        with col3:
            min_rating = st.slider("Minimum Rating:", 0.0, 5.0, 3.0, 0.5)
        
        # Filter destinations
        filtered = self.destinations.copy()
        
        if selected_type != "All":
            filtered = filtered[filtered['Type'] == selected_type]
        
        if selected_state != "All":
            filtered = filtered[filtered['State'] == selected_state]
        
        filtered = filtered[filtered['AvgRating'] >= min_rating]
        
        st.write(f"**Showing {len(filtered)} destinations**")
        
        # Display in a table
        display_cols = ['Name', 'State', 'Type', 'AvgRating', 'Popularity', 'BudgetLevel', 'Season']
        available_cols = [col for col in display_cols if col in filtered.columns]
        
        st.dataframe(
            filtered[available_cols].sort_values('Popularity', ascending=False),
            use_container_width=True,
            height=600
        )
    
    def render_feedback_tab(self):
        """Render feedback history tab"""
        st.header("📝 Your Feedback History")
        
        if not st.session_state.feedback_history:
            st.info("No feedback recorded yet. Start exploring destinations!")
            return
        
        feedback_df = pd.DataFrame(st.session_state.feedback_history)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Feedback", len(feedback_df))
        
        with col2:
            interested = len(feedback_df[feedback_df['feedback'] == 'interested'])
            st.metric("Interested", interested)
        
        with col3:
            not_interested = len(feedback_df[feedback_df['feedback'] == 'not_interested'])
            st.metric("Not Interested", not_interested)
        
        # Display feedback
        st.dataframe(feedback_df, use_container_width=True)
        
        if st.button("Clear Feedback History"):
            st.session_state.feedback_history = []
            st.rerun()
    
    def run(self):
        """Run the application"""
        self.render_header()
        
        # Sidebar
        preferences = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Recommendations",
            "📊 Statistics",
            "🗺️ Explore",
            "📝 Feedback"
        ])
        
        with tab1:
            self.render_recommendations()
        
        with tab2:
            self.render_statistics()
        
        with tab3:
            self.render_explore_tab()
        
        with tab4:
            self.render_feedback_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: #666;'>"
            "Virtual Travel Agent | Powered by AI | © 2025"
            "</p>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    app = TravelAgentApp()
    app.run()