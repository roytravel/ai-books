from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import requests
from rasa_core_sdk import Action
from rasa_core_sdk.events import SlotSet

class GetTodaysHoroscope(Action):
    def name(self):
        return "get_todays_horoscope"

    def run(self, dispatcher, tracker, domain):
        user_horoscope_sign = tracker.get_slot('horoscope_sign')
        # base_url = "http://horoscope-api.herokuapp.com/horoscope/{day}/{sign}"
        base_url = "http://localhost:5000/horoscope/{day}/{sign}"
        url = base_url.format(**{'day':"today", 'sign':user_horoscope_sign})
        res = requests.get(url).text
        print (res)
        # todays_horoscope = res.json()['horoscope']
        # response = "Your today's horoscope:\n{}".format(todays_horoscope)
        dispatcher.utter_message(res)
        return [SlotSet("horoscope_sign", user_horoscope_sign)]

class SubscribeUser(Action):
    def name(self):
        return "subscribe_user"
    
    def run(self, dispatcher, tracker, domain):
        
        subscribe = tracker.get_slot('subscribe')

        if subscribe == "True":
            response = "You're successfully subscribed"
        if subscribe == "False":
            response = "Your're successfully unsubscribed"
        
        dispatcher.utter_message(response)
        return [SlotSet("subscribe", subscribe)]