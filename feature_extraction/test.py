import scipy.io.wavfile as wave

FILE = "spring.wav"
(rate, rawData) = wave.read(FILE)
WIN = int(rate * 0.016)
STEP = int(rate * 0.016)
LEN = rawData.shape[0]

curPos = 0
rawDataNew = numpy.array([], dtype=numpy.float64)

while (curPos + WIN - 1 < LEN):

