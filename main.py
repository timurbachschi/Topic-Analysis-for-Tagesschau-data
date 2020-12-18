from speech2text import transcribe_from_daterange
from get_video import download_videos_in_timeperiod

def main():
    #22.10.
    transcribe_from_daterange("20181022", "20181129", download=True)
    #download_videos_in_timeperiod("20190614", "20201130")

if __name__ == "__main__":
    main()
