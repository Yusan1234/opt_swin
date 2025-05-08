# main.py
from train import train_swin

if __name__ == '__main__':
    prof = train_swin(batch_size=32, epochs=1)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))