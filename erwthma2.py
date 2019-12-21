

# erwthma 2
# a meros

bits = 5
levels = 2 ** bits-1
delta = 4.0 / levels # απόσταση μεταξύ των levels
fm = 9000
fs = fm * 45
A = 4
time1 = sample_time(fm, fs, 4)
y = signal_sq_triangle(A, fm, time1)
y_quantized = np.zeros(len(y))
h = np.zeros(len(y))
for i in range(0, len(y)):
    h[i] = 2 * (y[i] // delta) + 1
    y_quantized[i] = (delta / 2) * h[i]  # mid riser, y_quantized = delta*([y/delta]+0.5)
y_axis = np.linspace(-4 + delta / 2, 4 - delta / 2, 31)
G_C = GrayCode(5)
grid(True)
plt.yticks(y_axis, G_C.generate_gray())

plt.stem(time1, y_quantized)
plt.xlabel("Time (sec)")
plt.ylabel("Quantizing Levels ( Natural Binary Coding 4bit )")
plt.title("Quantized Signal")
plt.show()

#b meros
def SNR(y, y_quantized, length):
    # η συνάρτηση υπολογίζει το Quantum error
    quantization_error = y_quantized - y
    average_error = 0
    for i in range(length):
        average_error += quantization_error[i]/length
    error_power = 0
    for i in range(length):
        error_power += ((quantization_error[i]-average_error)**2) / length
    print("Standard Deviation error for", length, "samples:", np.sqrt(error_power))
    signal_power = 0
    for i in range(length):
        signal_power += ((y[i])**2)/length
    SNR_exp = signal_power / error_power
    SNR_exp_dB = 10 * np.log10(SNR_exp)
    print("SNR for", length, "samples:", SNR_exp, "=", SNR_exp_dB, "dB\n")


SNR(y, y_quantized, 10)  # 10 samples
SNR(y, y_quantized, 20)  # 20 samples

#Theoritical SNR
var_error_theory = delta**2 / 12
signal_power_theory = sum(y**2)/len(y)
SNR_theory = (signal_power_theory/var_error_theory)
SNR_theory_dB = 10*np.log10(SNR_theory)
print ("Theoretical SNR: ", SNR_theory, "=", SNR_theory_dB, "dB" )
#erwthma c

G_C = G_C.generate_gray()
gray_list = list(G_C)
levels = np.rint(y_quantized/delta) #υπολογίζουμε το level στο οποίο βρίσκεται το i-οστο κβάντο
levels = [int(x) for x in levels]
gray_function = [0]*(np.size(levels))
for i in range(np.size(levels)):
    gray_function[i] = int(gray_list[levels[i]-1]) # η συνάρτηση gray_function περιέχει το level στο οποίο βρίσκεται το
       #i-οστο κβάντο σε κώδικα gray
sign = 1
bit_stream = [0]*(10*np.size(gray_function))
for i in range(np.size(gray_function)):
    for j in range(5):
        bit_stream[10*i+2*j] = (gray_function[i]//10**(4-j))%2*sign*9 #Η bit_stream έχει τα bits που προκύπτουν από την
          #gray_με κωδικοποίση γραμμής BIPOLAR RZ
        if ((gray_function[i]//10**(4-j))%2 == 1):
            sign = sign*(-1)
        bit_stream[10*i+2*j+1] = 0
t = np.linspace(0, 0.0005*50, 50*100) #θα παρουσιάσουμε τα πρώτα 25 bits
sig_transmitted = [0]*np.size(t)
for i in range(np.size(t)):
    sig_transmitted[i] = bit_stream[i//100]
plt.figure(figsize=(20, 15))
plt.plot(t, sig_transmitted)
plt.grid()
plt.xlabel ("Time(s)")
plt.ylabel ("Voltage(V)")
plt.title("Bit stream for the first 25 bits")
plt.show()
