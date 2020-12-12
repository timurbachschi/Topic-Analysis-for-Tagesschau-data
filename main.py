from speech2text import transcribe_from_daterange
from get_video import download_videos_in_timeperiod

def main():
    transcribe_from_daterange("20191031", "20200102")
    #download_videos_in_timeperiod("20190614", "20201130")

if __name__ == "__main__":
    main()
