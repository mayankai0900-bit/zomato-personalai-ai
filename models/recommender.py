
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

customers = pd.read_csv("data/customers.csv")
restaurants = pd.read_csv("data/restaurants.csv")
orders = pd.read_csv("data/orders.csv")

def minmax(x):
    x=np.asarray(x,dtype=float)
    if x.max()==x.min():
        return np.ones_like(x)*0.5
    return (x-x.min())/(x.max()-x.min())

def recommend(customer_id, max_budget=600, cuisine=None, spicy=None, new_only=False, top_n=5):
    c=customers[customers.customer_id==customer_id].iloc[0]
    rs=restaurants[restaurants.city==c.city].copy()

    # Hard filters
    rs=rs[rs.average_order_value <= max_budget]
    if cuisine:
        rs=rs[(rs.cuisine.str.lower()==cuisine.lower()) | (rs.secondary_cuisine.str.lower()==cuisine.lower())]
    if spicy is True:
        rs=rs[rs.spice_score>=0.55]

    # Remove restaurants already tried when user requests novelty
    if new_only:
        tried=set(orders.loc[orders.customer_id==customer_id,"restaurant_id"])
        rs=rs[~rs.restaurant_id.isin(tried)]

    if rs.empty:
        return rs

    # Customer preference features
    cuisine_match=np.where(
        rs.cuisine.eq(c.preferred_cuisine),1.0,
        np.where(rs.cuisine.eq(c.secondary_cuisine),0.6,0.15)
    )
    budget_match=1-np.minimum(np.abs(rs.average_order_value-c.avg_order_value)/max(c.avg_order_value,1),1)
    healthy_match=(rs.healthy_score*c.healthy_food_affinity) + (1-c.healthy_food_affinity)*0.5
    spicy_match=(rs.spice_score*c.spicy_food_affinity) + (1-c.spicy_food_affinity)*0.5
    quality=minmax(rs.rating)
    speed=1-minmax(rs.delivery_time_min)
    exploration=rs.popularity_score*(1-c.restaurant_loyalty) + rs.new_restaurant*(c.new_restaurant_affinity)

    rs["preference_score"]=0.30*cuisine_match
    rs["behaviour_score"]=0.20*(0.55*healthy_match+0.45*spicy_match)
    rs["context_score"]=0.15*speed
    rs["quality_score"]=0.15*quality
    rs["value_score"]=0.10*budget_match
    rs["exploration_score"]=0.10*exploration

    rs["final_score"]=(rs["preference_score"]+rs["behaviour_score"]+
                       rs["context_score"]+rs["quality_score"]+
                       rs["value_score"]+rs["exploration_score"])
    return rs.sort_values("final_score",ascending=False).head(top_n)

# Example:
# print(recommend("C0001", max_budget=500, spicy=True, new_only=True))
