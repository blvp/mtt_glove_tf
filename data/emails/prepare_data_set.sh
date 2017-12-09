rm -rf raw
mkdir -p raw/spam
mkdir -p raw/ham
ls -l *.tar.gz | awk '{print $9}' | xargs -I {} sh -c 'tar xzvf {}'
cp -R enron*/* raw/
rm -rf enron*/
