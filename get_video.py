"""Module for downloading the episodes of tagesschau for further analysis in the project"""
import re
import urllib.request
import subprocess
import os
from pathlib import Path
import requests as req
from bs4 import BeautifulSoup
from datetime import date, datetime
import pandas as pd



def mp4_to_mp3(in_path, out_path):
    # Converts mp4 to mp3 using ffmpeg
    command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(in_path, out_path)
    subprocess.call(command, shell=True)


def get_video_urls_by_date(date_str):
    # Return set of all URls from given date (encoded as "yyyymmdd")
    url = "https://www.tagesschau.de/multimedia/video/videoarchiv2~_date-{}.html".format(date_str)
    resp = req.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    video_urls = set()
    for anchor in soup.find_all("a", href=re.compile(r'\/multimedia\/sendung\/.+')):
        video_urls.add("https://www.tagesschau.de/" + anchor["href"])
    return video_urls


def download_video(path):
    # Download tagesschau episode from given path.
    resp = req.get(path)
    soup = BeautifulSoup(resp.text, 'html.parser')
    download_link = soup.find_all("a", href=re.compile(r'.*h264.mp4'))[0]["href"]
    episode_title = "_".join(soup.find("h1").text.split(" "))[7:].replace(".", "").replace(":", "").replace("Uhr", "")
    urllib.request.urlretrieve("https:" + download_link, 'tagesschau.mp4')
    return episode_title


def download_videos_by_date(date_str):
    """
    Downloads all tagesschau episodes for a given date.

            Parameters:
                    date_str (str): String encoding of the date (format "yyyymmdd")

            Returns:
                    None
    """
    urls = get_video_urls_by_date(date_str)
    for url in urls:
        episode_title = download_video(url)
        Path("episodes").mkdir(parents=True, exist_ok=True)
        out_path = "episodes/" + episode_title + ".mp3"
        mp4_to_mp3("tagesschau.mp4", out_path)
        os.remove("tagesschau.mp4")


def download_videos_in_timeperiod(start_date, end_date):
    """
    Downloads all tagesschau episodes for a given time period.

            Parameters:
                    start_date (datetime): Start date for the time period
                    end_date (datetime): End date for the time period

            Returns:
                    None
    """
    daterange = pd.date_range(start_date, end_date)
    for date in daterange:
        download_videos_by_date(date.strftime("%Y%m%d"))
