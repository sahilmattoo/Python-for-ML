# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 19:34:37 2021

@author: smattoo5
"""

train_split_size = .2
straity_flag = True
strategy = "median"


drop_x = ['Satisfaction', 'DepartureDelayin_Mins']
col_y = ['Satisfaction']

map1 = {'extremely poor' : 0, 'poor' : 1, 'need improvement' : 2,
                'acceptable' : 3,'good' : 4, 'excellent' : 5, 'not_captured' : 2}

map2 = {'very inconvinient' : 0, 'Inconvinient' : 1, 'need improvement' : 2, 
                'manageable' : 3,'Convinient' : 4, 'very convinient' : 5}

map3 = {'Loyal Customer' : 1, 'disloyal Customer' : 0,'Business travel' : 1, 
                'Personal Travel' : 0,'Female' : 0, 'Male' : 1,'satisfied' : 1, 
                'neutral or dissatisfied' : 0, 'Eco Plus': 0 , 'Eco': 1, 
                'Business': 2}


numeric_features = ['Age', 'Flight_Distance', 'DepartureDelayin_Mins']

feedback_features = ['Seat_comfort', 'Departure.Arrival.time_convenient', 'Food_drink',
       'Gate_location', 'Inflightwifi_service', 'Inflight_entertainment',
       'Online_support', 'Ease_of_Onlinebooking', 'Onboard_service',
       'Leg_room_service', 'Baggage_handling', 'Checkin_service',
       'Cleanliness', 'Online_boarding']

other_cat_cols =  ['Gender', 'CustomerType', 'TypeTravel', 'Class']

