import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_ab_test_data(n_users=50000, seed=42, output_path='ab_test_data.csv'):
    """
    Generates synthetic user-level and funnel-step event data for an e-commerce One-Click Checkout A/B test case study.
    
    Experiment Simulation Parameters:
    - 50,000 synthetic users randomly assigned 50/50 to:
      * 'control': Legacy 3-Step Checkout Flow
      * 'treatment': One-Click Checkout Redesign
    - 4 Pre-Planned User Segments (hardcoded target conversion rates):
      * 'new_user' (30%): First-time buyers. High sensitivity to one-click UX (+26% lift engineered).
      * 'mobile' (30%): Mobile device sessions. Reduced form-filling friction (+24% lift engineered).
      * 'desktop' (20%): Desktop users. Moderate lift (+14% lift engineered).
      * 'power_user' (20%): Frequent buyers. Modeled with an engineered drop (-8% lift) to simulate potential friction.
        Note: The narrative regarding saved cards/addresses is a motivating design hypothesis; this dataset does
        not measure address changes or payment preferences.
    - 4 Countries: US (45%), UK (25%), DE (15%), IN (15%)
    - 5 Funnel Steps:
      * 1: Landing
      * 2: Cart
      * 3: Checkout
      * 4: Payment
      * 5: Purchase
    - Guardrail Metric:
      * order_value (AOV) is log-normally distributed, generated without an engineered treatment decline.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # 50/50 random variant split
    variants = np.random.choice(['control', 'treatment'], n_users, p=[0.50, 0.50])
    
    # Segment distribution
    segments = np.random.choice(
        ['new_user', 'power_user', 'mobile', 'desktop'], 
        n_users, p=[0.30, 0.20, 0.30, 0.20]
    )
    
    # Geographic distribution
    countries = np.random.choice(['US', 'UK', 'DE', 'IN'], n_users, p=[0.45, 0.25, 0.15, 0.15])
    
    # 28-day experiment window
    base_date = datetime.now() - timedelta(days=28)
    day_offsets = np.random.randint(0, 28, n_users)
    second_offsets = np.random.randint(0, 86400, n_users)
    
    # Pre-calibrated cumulative step completion probabilities: [Landing, Cart, Checkout, Payment, Purchase]
    # Ensures exact target conversion rates and step drops
    cumulative_rates = {
        ('control', 'new_user'):     [1.000, 0.510, 0.285, 0.170, 0.100],
        ('treatment', 'new_user'):   [1.000, 0.570, 0.340, 0.215, 0.126], # +26.0% lift (sig)
        
        ('control', 'mobile'):       [1.000, 0.560, 0.345, 0.230, 0.155],
        ('treatment', 'mobile'):     [1.000, 0.630, 0.420, 0.290, 0.192], # +23.9% lift (sig)
        
        ('control', 'desktop'):      [1.000, 0.640, 0.430, 0.295, 0.195],
        ('treatment', 'desktop'):    [1.000, 0.675, 0.470, 0.335, 0.222], # +13.8% lift (sig)
        
        ('control', 'power_user'):   [1.000, 0.795, 0.640, 0.500, 0.375],
        ('treatment', 'power_user'): [1.000, 0.770, 0.600, 0.455, 0.345], # -8.0% lift (statistically significant friction)
    }
    
    rows = []
    
    for i in range(n_users):
        variant = variants[i]
        segment = segments[i]
        country = countries[i]
        timestamp = base_date + timedelta(days=int(day_offsets[i]), seconds=int(second_offsets[i]))
        user_id = f"U_{i:06d}"
        session_id = f"S_{i:06d}"
        
        cum_probs = cumulative_rates[(variant, segment)]
        
        # Determine maximum funnel step reached by this user
        # User survives to step k if random < cum_probs[k-1]
        user_rand = random.random()
        max_completed_step = 1
        for step_idx in range(1, 5): # check steps 2, 3, 4, 5
            if user_rand < cum_probs[step_idx]:
                max_completed_step = step_idx + 1
            else:
                break
                
        # Base session duration (seconds)
        base_dur = random.randint(35, 320)
        
        for step in range(1, 6):
            completed = (step <= max_completed_step)
            
            order_value = None
            if step == 5 and completed:
                # Log-normal distribution for realistic e-commerce order value
                # Guardrail: control vs treatment average order values remain statistically comparable
                if segment == 'power_user':
                    val = np.random.lognormal(mean=4.12, sigma=0.62)
                elif segment == 'new_user':
                    val = np.random.lognormal(mean=3.72, sigma=0.72)
                else:
                    val = np.random.lognormal(mean=3.88, sigma=0.68)
                order_value = round(float(val), 2)
                base_dur += random.randint(45, 180)
            
            rows.append({
                'user_id': user_id,
                'session_id': session_id,
                'variant': variant,
                'user_segment': segment,
                'country': country,
                'funnel_step': step,
                'step_completed': completed,
                'timestamp': timestamp,
                'order_value': order_value,
                'session_duration_sec': base_dur
            })
            
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Generated {len(df):,} event rows for {df['user_id'].nunique():,} unique users.")
    print(f"Saved to: {output_path}")
    return df

if __name__ == '__main__':
    generate_ab_test_data()
