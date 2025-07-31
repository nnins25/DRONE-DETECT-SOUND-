import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# -----------------------------------------------------------------------------
# 1단계: 가상 마이크 어레이 데이터 생성 (이전과 동일)
# -----------------------------------------------------------------------------
def generate_virtual_mic_data(
    source_angle_deg=30, drone_freq=800, num_channels=8, mic_distance=0.05,
    duration=1.0, samplerate=44100, noise_level=0.3
):
    """가상 마이크 어레이 데이터를 생성하는 함수입니다."""
    SPEED_OF_SOUND = 343.0
    NUM_SAMPLES = int(samplerate * duration)
    mic_positions = np.arange(num_channels) * mic_distance
    mic_positions -= np.mean(mic_positions)
    source_angle_rad = np.deg2rad(source_angle_deg)
    # t = np.linspace(0., duration, NUM_SAMPLES, endpoint=False) # Not needed for return
    source_signal = np.sin(2 * np.pi * drone_freq * np.linspace(0., duration, NUM_SAMPLES, endpoint=False))
    noise_signal = np.random.randn(NUM_SAMPLES, num_channels) * noise_level
    multi_channel_data = np.zeros((NUM_SAMPLES, num_channels))
    for i in range(num_channels):
        mic_pos = mic_positions[i]
        tau = (mic_pos * np.sin(source_angle_rad)) / SPEED_OF_SOUND
        sample_shift = int(round(tau * samplerate))
        multi_channel_data[:, i] = np.roll(source_signal, sample_shift)
    final_audio_chunk = multi_channel_data + noise_signal
    return final_audio_chunk, samplerate, mic_positions

# -----------------------------------------------------------------------------
# 2단계: 신호 전처리 (이전과 동일)
# -----------------------------------------------------------------------------
def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """오디오 데이터에 대역 통과 필터를 적용합니다."""
    if fs <= 0: raise ValueError("Sampling rate must be positive.")
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

# -----------------------------------------------------------------------------
# 3단계: 음원 위치 추정 (이전과 동일)
# -----------------------------------------------------------------------------
def delay_and_sum_beamforming(data, mic_positions, fs):
    """딜레이-앤-섬 빔포밍을 수행하여 음원의 방향을 추정합니다."""
    SPEED_OF_SOUND = 343.0
    num_samples, num_channels = data.shape
    angles_to_scan = np.deg2rad(np.arange(-90, 91))
    power_map = np.zeros(len(angles_to_scan))
    for i, angle in enumerate(angles_to_scan):
        summed_signal = np.zeros(num_samples)
        for ch in range(num_channels):
            tau = (mic_positions[ch] * np.sin(angle)) / SPEED_OF_SOUND
            sample_shift = int(round(tau * fs))
            summed_signal += np.roll(data[:, ch], -sample_shift)
        power_map[i] = np.sum(summed_signal**2)
    return power_map, angles_to_scan

# -----------------------------------------------------------------------------
# 4단계: 시각화 (레이더 스크린)
# -----------------------------------------------------------------------------
def plot_radar_screen(power_map, angles_rad, detected_angle_deg, true_angle_deg):
    """
    계산된 음향 맵을 레이더 스크린 형태로 시각화합니다.

    Args:
        power_map (np.array): 각도별 음향 파워 배열.
        angles_rad (np.array): 스캔한 각도 배열 (라디안).
        detected_angle_deg (float): 탐지된 최종 각도 (도).
        true_angle_deg (float): 실제 음원의 각도 (도, 비교용).
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 1. 레이더 화면의 기본 설정을 구성합니다.
    ax.set_theta_zero_location('N')  # 0도를 북쪽(위)으로 설정
    ax.set_theta_direction(-1)       # 각도를 시계 방향으로 증가
    ax.set_thetalim(-np.pi/2, np.pi/2) # -90도(왼쪽)부터 +90도(오른쪽)까지만 표시
    ax.set_yticklabels([])           # 반지름 눈금(파워 레벨)은 숨깁니다.
    ax.set_rlim(0, np.max(power_map) * 1.1) # 파워 최대치에 맞춰 반지름 범위 설정
    
    # 2. 음향 맵 데이터를 레이더에 그립니다.
    ax.plot(angles_rad, power_map, color='dodgerblue', linewidth=2)
    # 파워가 강한 영역을 채워서 강조합니다.
    ax.fill(angles_rad, power_map, color='dodgerblue', alpha=0.3)
    
    # 3. 탐지된 방향을 빨간색 선으로 명확하게 표시합니다.
    ax.plot([np.deg2rad(detected_angle_deg), np.deg2rad(detected_angle_deg)],
            [0, np.max(power_map)],
            color='red', linestyle=':', linewidth=2,
            label=f'Detected: {detected_angle_deg:.1f}°')
            
    # 4. 실제 방향을 초록색 선으로 표시하여 비교합니다.
    ax.plot([np.deg2rad(true_angle_deg), np.deg2rad(true_angle_deg)],
            [0, np.max(power_map)],
            color='green', linestyle='--', linewidth=2,
            label=f'True: {true_angle_deg}°')
    
    ax.set_title("Acoustic Radar Screen", fontsize=16, y=1.1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- 설정 ---
    TRUE_ANGLE_DEG = -25  # 실제 드론의 위치를 -25도(왼쪽)로 설정

    # --- 1, 2, 3단계 순차 실행 ---
    unfiltered_audio, SAMPLERATE, mic_positions = generate_virtual_mic_data(source_angle_deg=TRUE_ANGLE_DEG)
    filtered_audio = apply_bandpass_filter(unfiltered_audio, 500.0, 2000.0, SAMPLERATE)
    power_map, angles_rad = delay_and_sum_beamforming(filtered_audio, mic_positions, SAMPLERATE)
    
    # --- 결과 분석 ---
    detected_angle_idx = np.argmax(power_map)
    detected_angle_deg = np.rad2deg(angles_rad[detected_angle_idx])
    
    print(f"\n>>> True Angle: {TRUE_ANGLE_DEG}°")
    print(f">>> Detected Angle: {detected_angle_deg:.1f}°")

    # --- 4단계 실행 ---
    plot_radar_screen(power_map, angles_rad, detected_angle_deg, TRUE_ANGLE_DEG)