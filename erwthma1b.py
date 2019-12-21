import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from pylab import *

#erwthma 1a
fm  = 9000

A = 4  #Amplitude

def sample_time(f, fs, Tnum):
    # Διάνυσμα χρόνου μέχρι <Tnum> περιόδους με δείγμα ανά 1/fs (sec)
    return np.arange(0, Tnum * 1/f, 1/fs)
def signal(A ,f ,t ):
    return A * pow(ss.sawtooth(2 * np.pi * f * t),2)
def signal_sinim(A ,f ,t ):
    return A*np.sin(2*np.pi*f*t)
#def plot_size(length, height):
    # Καθορισμός διαστάσεων διαγραμμάτων
 # return plt.figure(figsize=(length, height))


fm2 = 10000
fd = 1000  #frequency of q(t)
fd1=9500
fs1 = fd * 25
fs2 = fd * 60
fs3 = fd * 5

grid(True)
# Plot 1 period
# z = y + sin(2πt*fm2)
t = np.linspace(0, 0.001, 10000)
q = np.add(signal_sinim(1,fm,t), signal_sinim(1,fm2,t))
plt.plot(t, q)
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("q(t)")
plt.show()


grid(True)
# Plot 1 period
tz1 = sample_time(fd,fs1,1)
# z = y + sin(2πt*fm2)
z1  = np.add(signal_sinim(1,fm,tz1), signal_sinim(1,fm2,tz1))
plt.plot(tz1, z1, '.')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(i) Sampling with fs1")
plt.show()
grid(True)
tz2 = sample_time(fd,fs2,1)
z2  = np.add(signal_sinim(1,fm,tz2), signal_sinim(1,fm2,tz2))
plt.plot(tz2, z2, '.r')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(ii) Sampling with fs2")
plt.show()

grid(True)
plt.plot(tz2, z2, '.r', label='Sampling with fs2')
plt.plot(tz1, z1, '.',label='Sampling with fs1')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("(iii) Shared Diagram")
plt.legend(loc=1)
plt.show()
grid(True)
tz3 = sample_time(fd,fs3,1)
z3  = np.add(signal_sinim(1,fm,tz3), signal_sinim(1,fm2,tz3))
plt.plot(tz3, z3, 'o')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude (V)")
plt.title("Sampling with 5*fd")


plt.show()