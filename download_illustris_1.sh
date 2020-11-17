mkdir -p data/Illustris-3
mkdir -p data/Illustris-3/groups_135
cd data/Illustris/groups_135
wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --content-disposition --header="API-Key: INSERT API KEY" "http://www.illustris-project.org/api/Illustris-3/files/groupcat-135/?format=api"
