from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split