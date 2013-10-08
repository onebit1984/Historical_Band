import pickle
import unittest 
import datetime as dt
from build_hmm import Steam,Weather
from build_hmm import HMM
from gen_hist_band import HistBand

class HistBandTest(unittest.TestCase):
    '''
    Test class to test validity of data in Historical Band
    '''
    def setUp(self):
        self.hist_band_obj = HistBand()
        if is_pickled_model:
             

class HMMTest(unittest.TestCase):
    '''
    Test class to test validity of hmm model
    '''
    def setUp(self):
        self.hmm_obj = HMM()

    def test_model_length(self):
        hmm_obj = self.hmm_obj
        self.assertTrue(len(hmm_obj)==len(hmm_obj.steam_obj))

    def test_hidden_states(self):
        '''
        test if all states are present in hidden states
        '''
        hmm_obj = self.hmm_obj
        hmm_obj.build_model()
        states = set(hmm_obj.hidden_states)
        self.assertFalse(len(states.difference(set(range(hmm_obj.n_states)))))

class DataTest(unittest.TestCase):
    '''
    Test class to test data validity for weather and steam
    '''
    def setUp(self):
        self.steam_obj = Steam()
        self.weather_obj = Weather()

    def test_weather_data(self):
        self.assertTrue(len(self.weather_obj))

    def test_steam_data(self):
        self.assertTrue(len(self.steam_obj))

    def test_weather_data_date(self):
        #test the date range for weather data 
        weather_obj_test = Weather(dt.date(2012,2,1),dt.date(2013,7,7),filename='../Data/OBSERVED_WEATHER_AUG.csv')
        self.assertFalse(weather_obj_test.ts.get(dt.datetime(2012,1,1),0))
        self.assertFalse(weather_obj_test.ts.get(dt.datetime(2013,8,1),0))
    
    def test_steam_data_date(self):
        #test the date range for steam data 
        steam_obj_test = Steam(dt.date(2012,2,1),dt.date(2013,7,7))
        self.assertFalse(steam_obj_test.ts.get(dt.datetime(2012,1,1,0,0),0))
        self.assertFalse(steam_obj_test.ts.get(dt.datetime(2013,8,1,0,0),0))

    def test_weather_steam_lengths(self):
        #compare lengths of weather and steam objects 
        steam_obj_test = Steam(dt.date(2012,2,1),dt.date(2013,7,7),filename='../Data/RUDINSERVER_CURRENT_STEAM_DEMAND_FX70_AUG.csv')
        weather_obj_test = Weather(dt.date(2012,2,1),dt.date(2013,7,7),filename='../Data/OBSERVED_WEATHER_AUG.csv')
        self.assertTrue(len(weather_obj_test)==len(steam_obj_test))

    def test_weather_length(self):
        #test if weather data contains all values in the increments of 15 mins
        from_date = dt.date(2013,2,1)
        to_date = dt.date(2013,7,7)
        weather_obj_test = Weather(from_date,to_date,filename='../Data/OBSERVED_WEATHER_AUG.csv')
        time_delta =  to_date - from_date
        length = time_delta.total_seconds()/900
        self.assertTrue(length==len(weather_obj_test))

    def test_steam_length(self):
        #test if steam data contains all values in the increments of 15 mins
        from_date = dt.date(2013,2,1)
        to_date = dt.date(2013,7,7)
        steam_obj_test = Steam(from_date,to_date,filename='../Data/RUDINSERVER_CURRENT_STEAM_DEMAND_FX70_AUG.csv')
        time_delta =  to_date - from_date
        length = time_delta.total_seconds()/900
        self.assertTrue(length==len(steam_obj_test))

def main():
    #suite = unittest.TestSuite()
    #suite.addTest(DataTest("test_weather_steam_lengths"))
    suite = unittest.TestLoader().loadTestsFromTestCase(HMMTest)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__=='__main__':
    main()
