import os
import re

import requests


def normalize_text(text: str):
    text = re.sub(r'[′’]', '\'', text)
    text = re.sub(r'\'\'', ' ', text)
    text = re.sub(r'\'', ' \' ', text)
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'"', ' " ', text)
    text = re.sub(r'\.', ' . ', text)
    text = re.sub(r'<br\s*/>', ' ', text)
    text = re.sub(', ', ' , ', text)
    text = re.sub('\(', ' ( ', text)
    text = re.sub('\)', ' ) ', text)
    text = re.sub('!', ' ! ', text)
    text = re.sub('\?', ' ? ', text)
    text = re.sub(';', ' ', text)
    text = re.sub(':', ' ', text)
    text = re.sub('-', ' - ', text)
    text = re.sub('=', ' ', text)
    text = re.sub('\*', ' ', text)
    text = re.sub('\|', ' ', text)
    text = re.sub('«', ' ', text)
    text = re.sub(r'[0-9]+', ' ', text)
    return text


"""

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

NOW=$(date +"%Y%m%d")

ROOT="data/wikimedia/${NOW}"
mkdir -p "${ROOT}"
echo "Saving data in ""$ROOT"
read -r -p "Choose a language (e.g. en, bh, fr, etc.): " choice
LANG="$choice"
echo "Chosen language: ""$LANG"
read -r -p "Continue to download (WARNING: This might be big and can take a long time!)(y/n)? " choice
case "$choice" in
  y|Y ) echo "Starting download...";;
  n|N ) echo "Exiting";exit 1;;
  * ) echo "Invalid answer";exit 1;;
esac
wget -c "https://dumps.wikimedia.org/""$LANG""wiki/latest/""${LANG}""wiki-latest-pages-articles.xml.bz2" -P "${ROOT}"
echo "Processing ""$ROOT"/"$LANG""wiki-latest-pages-articles.xml.bz2"
bzip2 -c -d "$ROOT"/"$LANG""wiki-latest-pages-articles.xml.bz2" | awk '{print tolower($0);}' | perl -e '
# Program to filter Wikipedia XML dumps to "clean" text consisting only of lowercase
# letters (a-z, converted from A-Z), and spaces (never consecutive)...
# All other characters are converted to spaces.  Only text which normally appears.
# in the web browser is displayed.  Tables are removed.  Image captions are.
# preserved.  Links are converted to normal text.  Digits are spelled out.
# *** Modified to not spell digits or throw away non-ASCII characters ***
# Written by Matt Mahoney, June 10, 2006.  This program is released to the public domain.
$/=">";                     # input record separator
while (<>) {
  if (/<text /) {$text=1;}  # remove all but between <text> ... </text>
  if (/#redirect/i) {$text=0;}  # remove #REDIRECT
  if ($text) {
    # Remove any text not normally visible
    if (/<\/text>/) {$text=0;}
    s/<.*>//;               # remove xml tags
    s/&amp;/&/g;            # decode URL encoded chars
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<ref[^<]*<\/ref>//g;  # remove references <ref...> ... </ref>
    s/<[^>]*>//g;           # remove xhtml tags
    s/\[http:[^] ]*/[/g;    # remove normal url, preserve visible text
    s/\|thumb//ig;          # remove images links, preserve caption
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/[[$1]]/ig;  # show categories without markup
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;  # remove links to other languages
    s/\[\[[^\|\]]*\|/[[/g;  # remove wiki url, preserve visible text
    s/{{[^}]*}}//g;         # remove {{icons}} and {tables}
    s/{[^}]*}//g;
    s/\[//g;                # remove [ and ]
    s/\]//g;
    s/&[^;]*;/ /g;          # remove URL encoded chars
    $_=" $_ ";
    chop;
    print $_;
  }
}
' | normalize_text | awk '{if (NF>1) print;}' | tr -s " " | shuf > "${ROOT}"/wiki."${LANG}".txt
"""


def download_wiki_dump(root, lang='en'):
    url = "https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2".format(lang=lang)
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    local_file_path = os.path.join(root, local_filename)
    with open(local_file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                # f.flush() commented by recommendation from J.F.Sebastian
    return local_file_path


if __name__ == '__main__':
    download_wiki_dump(os.getcwd())
