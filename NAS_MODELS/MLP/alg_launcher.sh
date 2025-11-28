graph_size=(10 12 15 20 25)
for node_size in "${graph_size[@]}"
do
  if [ $node_size -eq 10 ]; then
    cd $node_size/Release/
  else
    cd ../../$node_size/Release/
  fi
  echo 100000000000 > fitness.txt
  rm Models/*
  rm results/*
  chmod +x ml_launcher.sh
  make P3
  ./P3 config/default.cfg config/p3.cfg
done

