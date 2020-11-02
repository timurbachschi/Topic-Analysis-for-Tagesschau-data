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


def transform_episode(in_path, out_path):
    # Transform one episode to specified dataformat (as given by the out path)
    _, data_format = os.path.splitext(out_path)
    directory = "episodes_{}".format(data_format)
    Path(directory).mkdir(parents=True, exist_ok=True)
    command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(in_path, out_path)
    subprocess.call(command, shell=True)


def transform_episodes(dir_path, data_format):
    # Converts all episodes (in mp4) in one directory to given format (mp3 etc.) using ffmpeg
    for filename in os.listdir(dir_path):
        in_path = "{}/{}".format(dir_path, filename)
        out_path = "{}/{}".format(directory, filename.replace("mp4", data_format))
        transform_episode(in_path, out_path)


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
    anchor_elems = soup.find_all("a", href=re.compile(r'.*h264.mp4'))
    Path("episodes_mp4").mkdir(parents=True, exist_ok=True)
    if len(anchor_elems) > 0:
        download_link = anchor_elems[0]["href"]
        episode_title = "_".join(soup.find("h1").text.split(" "))[7:].replace(".", "").replace(":", "").replace("Uhr", "")
        urllib.request.urlretrieve("https:" + download_link, 'episodes_mp4/{}.mp4'.format(episode_title))
        return episode_title
    else:
        return None


def download_videos_by_date(date_str, transform=None):
    """
    Downloads all tagesschau episodes for a given date.

            Parameters:
                    date_str (str): String encoding of the date (format "yyyymmdd")
                    transform (str): Transform downloaded videos to given format, None by default.

            Returns:
                    None
    """
    urls = get_video_urls_by_date(date_str)
    for url in urls:
        episode_title = download_video(url)
        if transform:
            out_dir = "episodes_{}".format(transform)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            in_path = "episodes_mp4/{}.mp4".format(episode_title)
            out_path = "{}/{}.{}".format(out_dir, episode_title, transform)
            transform_episode(in_path, out_path)


def download_videos_in_timeperiod(start_date, end_date, transform=None):
    """
    Downloads all tagesschau episodes for a given time period.

            Parameters:
                    start_date (datetime): Start date for the time period
                    end_date (datetime): End date for the time period
                    transform (str): Transform downloaded videos to given format, None by default

            Returns:
                    None
    """
    daterange = pd.date_range(start_date, end_date)
    for date in daterange:
        download_videos_by_date(date.strftime("%Y%m%d"), transform)

#download_videos_by_date("20201028", "mp3")
