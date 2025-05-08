import torch
import numpy
import cmath
import math

def print_tensor(tensor):
    print("")
    print("LIST(", end=" ")
    print(*numpy.around(tensor.flatten().tolist(),decimals=6).tolist(), sep=", ", end=" ")
    print("),")

def print_tensor_shape(tensor):
    print("LIST(", end=" ")
    print(*list(tensor.size()), sep=", ", end=" ")
    print("),")

def irdft(X, N):
    K=len(X)
    N_even = N%2==0
    if N_even:
        K=K-1
    x=numpy.zeros(N, dtype=complex)
    
    for n in range(N):
        x[n] = X[0]
        for k in range(1,K): 
            x[n] = x[n] + (X[k]*numpy.exp(1j*2*cmath.pi*k*n/N))
            #print(f"X_k={k}, twiddle_k={k}")
            x[n] = x[n] + (numpy.conj(X[k])*numpy.exp(1j*2*cmath.pi*(N-k)*n/N))
            #print(f"X_k={k}, twiddle_k={N-k}")  

        if N_even:
            x[n] = x[n] + (X[K]*numpy.exp(1j*2*cmath.pi*(K)*n/N))
  
    return x.real*(1/N)

def istft_irfft(stft_matrix, n_fft, istft_window, hop_length=None, win_length=None, normalized=False, center = False):
    device = stft_matrix.device

    n_frames = stft_matrix.shape[-1]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    
    y = torch.zeros(expected_signal_len, device=device)
    sum_wind = torch.zeros(expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, i]
        iffted = torch.from_numpy(irdft(spec, n_fft)).to(device)

        sum_wind[sample:(sample+n_fft)] += istft_window**2
        y[sample:(sample+n_fft)] += iffted*istft_window

    scale: float = 1.0
    if normalized:
        scale = math.sqrt(frame_size)

    out = (y * scale) / sum_wind

    if center:
        margin = n_fft // 2
        out = out[margin:-margin]

    return out

#-----------------------------------------------------

def calcDivisor(idx: int, frameSize: int, frameStep: int, bufferSize: int):
    earliestStartIdx: int = max(0,idx-frameSize+1)
    realStartIdx: int = int((earliestStartIdx+frameStep-1)/frameStep)*frameStep

    lastPossibleIdx = bufferSize-frameSize

    if(lastPossibleIdx<realStartIdx):
        return 0

    ret: int = int((min(lastPossibleIdx,idx)-realStartIdx)/frameStep) + 1
    return ret

#-----------------------------------------------------

torch.manual_seed(564576)

batch = 2
frames = 3
center=True
normalized=True
frame_step = 1
frame_size = 14
windows_size = 10
length = None
window = torch.hamming_window(windows_size)
input = torch.randn(batch, int(frame_size/2)+1, frames, dtype=torch.cfloat)
output = torch.istft(input, frame_size, window=window, win_length=windows_size, normalized=normalized, hop_length=frame_step, center=center, onesided=True, return_complex=False, length=length)

print_tensor_shape(torch.view_as_real(input))
print_tensor_shape(window)
print_tensor_shape(output)
print(frame_size,",")
print(frame_step,",")
print(center,",")
print(normalized,",")
print(length,",")
print_tensor(torch.view_as_real(input))
print_tensor(window)
print_tensor(output)

output2 = istft_irfft(input, frame_size, window, frame_step, windows_size, normalized=normalized, center=center)

print(output2)
print(torch.allclose(output, output2, atol=1e-4))

# print(":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

# frameSize: int = 16
# frameStep: int = 5
# bufferSize: int = 21

# for i in range(bufferSize): 
#     print(f"idx: {i}, divisior: {calcDivisor(i, frameSize, frameStep, bufferSize)}")
