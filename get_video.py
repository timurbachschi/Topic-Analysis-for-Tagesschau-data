"""Module for downloading the episodes of tagesschau for further analysis in the project"""
import re
import subprocess
import os
from pathlib import Path
import requests as req
from bs4 import BeautifulSoup
from datetime import date, datetime
import pandas as pd
import shlex
import xml.etree.ElementTree as ET
import urllib.request, json
import xmltodict
import urllib


def measure_values_for_loudnorm(command):
    # TODO: What does this do?
    pre_run_command = shlex.split(command)
    print(pre_run_command)
    process = subprocess.Popen(pre_run_command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)

    stdout, stderr = process.communicate()
    for line in stderr.split("\n"):
        if "Input Integrated" in line:
            measured_Input = re.search("-?[0-9]+.?[0-9]*", line)[0]
    for line in stderr.split("\n"):
        if "Input True Peak" in line:
            measured_True_Peak = re.search("-?[0-9]+.?[0-9]*", line)[0]
            print("mTP:", measured_True_Peak)
            if float(measured_True_Peak) > 0:
                measured_True_Peak = '0.0'
            if float(measured_True_Peak) < -9:
                measured_True_Peak = '-9.0'
            print("mTP nachher:", measured_True_Peak)
    for line in stderr.split("\n"):
        if "Input LRA" in line:
            measured_LRA = re.search("-?[0-9]+.?[0-9]*", line)[0]
    print(measured_Input, measured_True_Peak, measured_LRA)
    return measured_Input, measured_True_Peak, measured_LRA


def transform_episode(in_path, out_path):
    # Transform one episode to specified dataformat (as given by the out path).
    # Use FFMPEG for this.
    data_format = 'flac'
    directory = "episodes_{}".format(data_format)
    Path(directory).mkdir(parents=True, exist_ok=True)
    pre_run = "ffmpeg -i {} -ar 16000 -ab 160k -vn -af loudnorm=linear=true:print_format=summary -f null -".format(
        in_path)
    measured_Input, measured_True_Peak, measured_LRA = measure_values_for_loudnorm(
        pre_run)
    print("main run...")
    fh = open("NUL", "w")
    main_run = "ffmpeg -i {} -ar 16000 -ab 160k -vn -af loudnorm=I=-19:TP={}:LRA={}:measured_I={}:measured_TP={}:measured_LRA={}:linear=true:print_format=summary -y {}".format(
        in_path, measured_True_Peak, measured_LRA, measured_Input, measured_True_Peak, measured_LRA, out_path)
    print("main_run command:", main_run)
    subprocess.run(main_run, shell=True, stdout=fh, stderr=fh)
    fh.close()


def transform_episodes(dir_path, data_format):
    # Converts all episodes (in mp4) in one directory to given format (mp3 etc.) using ffmpeg
    for filename in os.listdir(dir_path):
        in_path = "{}/{}".format(dir_path, filename)
        out_path = "{}/{}".format(dir_path,
                                  filename.replace("mp4", data_format))
        transform_episode(in_path, out_path)


def get_video_urls_by_date(date_str):
    # Return set of all URls from given date (encoded as "yyyymmdd")
    url = "https://www.tagesschau.de/multimedia/video/videoarchiv2~_date-{}.html".format(
        date_str)
    resp = req.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    video_urls = set()
    for anchor in soup.find_all("a", href=re.compile(r'\/multimedia\/sendung\/.+')):
        video_urls.add("https://www.tagesschau.de/" + anchor["href"])
    return video_urls


def download_video(path):
    # Download tagesschau episode from given path.
    episode_not_available = None
    resp = req.get(path)
    soup = BeautifulSoup(resp.text, 'html.parser')
    anchor_elems = soup.find_all("a", href=re.compile(r'.*h264.mp4'))
    Path("episodes_mp4").mkdir(parents=True, exist_ok=True)
    if len(anchor_elems) > 0:
        download_link = anchor_elems[0]["href"]
        episode_title = "_".join(soup.find("h1").text.split(" "))[7:].replace(".", "").replace(":", "").replace("Uhr", "")
        # See if episode is actually accessible
        try:
            urllib.request.urlretrieve(
                "https:" + download_link, 'episodes_mp4/{}.mp4'.format(episode_title))
            episode_not_available = False
        except urllib.error.HTTPError:
            print("Episode not available")
            episode_not_available = True
        return episode_title, episode_not_available
    else:
        return None, episode_not_available


def download_videos_by_date(date_str, transform=None):
    """
    Downloads all tagesschau episodes for a given date.

            Parameters:
                    date_str (str): String encoding of the date (format "yyyymmdd")
                    transform (str): Transform downloaded videos to given format, None by default.

            Returns:
                    List of filenames for each episode.
    """
    urls = get_video_urls_by_date(date_str)
    episode_filenames = []
    for url in urls:
        episode_title, episode_not_available = download_video(url)
        print(episode_title, episode_not_available)
        if "tagesschau" in str(episode_title):
            if str(episode_title+'.mp4') not in os.listdir("transcripts/"):
                episode_filenames.append(str(episode_title+'.mp4'))
                if transform is not None and episode_title is not None:
                    out_dir = "episodes_{}".format('flac')
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    in_path = "episodes_mp4/{}.mp4".format(episode_title)
                    out_path = "{}/{}.{}".format(out_dir,
                                                 episode_title, 'flac')
                    if transform and episode_not_available == False:
                        transform_episode(in_path, out_path)
            else:
                print(episode_title, "is already downloaded")
    return episode_filenames


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
        print("Download episodes: {}".format(date))
        download_videos_by_date(date.strftime("%Y%m%d"), transform)


def get_mediadata_json(soup):
    # Gets mediadata in JSON format for an epidode loaded as a beautifulsoup object
    attrs = soup.find('form', {"class": "seperateFields"}).attrs
    for attr in attrs:
        if "id" in attr:
            video_id = attr.split("_")[1].replace("-entry", "")
            # Get URL of the tagesschau mediajson file that contains metadata
            mediajson_url = "https://www.tagesschau.de/multimedia/video/{}~mediajson.json".format(video_id)
            import urllib.request, json 
            with urllib.request.urlopen(mediajson_url) as json_url:
                json_data = json.loads(json_url.read().decode())
                return json_data
    print("Something went wrong!")
    return None


def get_subtitles_from_url(subtitle_url):
    # Return subtitles with timestamps based on URL for the subtitle XML found in the metadata JSON
    feed = urllib.request.urlopen(subtitle_url)
    tree = ET.parse(feed)
    root = tree.getroot()
    subtitles_list = []
    for child in root[1][0]:
        begin = child.attrib["begin"]
        end = child.attrib["end"]
        text = []
        for y in child.itertext():
            text.append(y)
        subtitles_list.append((begin, " ".join(text).replace("\n", "").replace("  ", ""), end))
    return subtitles_list


def get_additional_metadata_for_daterange(start_date, end_date):
        """
        Get subtitle and teaser text info of all episode in given date range ("%Y%m%d")

            Parameters:
                    start_date (datetime): Start date from which metadata shall be extracted
                    end_date (datetime): End date up to which metadata shall be extracted

            Returns:
                    Dictionary containing subtitles and "Themen der Sendung" for all episodes covered.
    """
    data_dict = {}
    daterange = pd.date_range(start_date, end_date)
    for date in daterange:
        date_str = date.strftime("%Y%m%d")
        urls = get_video_urls_by_date(date.strftime("%Y%m%d"))
        for url in urls:
            resp = req.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            episode_title = "_".join(soup.find("h1").text.split(" "))[7:].replace(".", "").replace(":", "").replace("Uhr", "")
            # Only get tagesschau episodes
            if "tagesschau" in episode_title and "vor" not in episode_title and "Gebärdensprache" not in episode_title:
                teaser = soup.findAll("p", {"class": "teasertext"})[0].get_text()
                ep_id = None
                # Get metadata first to find URL of subtitle XML
                try:
                    json_data = get_mediadata_json(soup)
                    if "_subtitleUrl" in json_data:
                        subtitle_url = "https://www.tagesschau.de{}".format(json_data["_subtitleUrl"])
                        try:
                            subtitle_list = get_subtitles_from_url(subtitle_url)
                            subtitle_text = "".join(filter(None, [x[1] for x in subtitle_list]))
                        # Subtitle XML does not exist
                        except urllib.error.HTTPError:
                            data_dict[date_str] = (teaser, None)
                        except ET.ParseError:
                            data_dict[date_str] = (teaser, None)
                    else:
                        subtitle_text = None
                    data_dict[date_str] = (teaser, subtitle_text)
                except urllib.error.HTTPError:
                    continue
    return data_dict