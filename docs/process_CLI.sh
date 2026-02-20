echo "==================================" >> mem_log.txt
echo "=== Process Cold Phase ===" >> mem_log.txt
date >> mem_log.txt

uname -a >> mem_log.txt
python3 --version >> mem_log.txt

for i in {1..10}
do
  echo "--- Process Cold Run $i ---" >> mem_log.txt
  
  sudo sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
  sleep 1
  
  echo "[Before Inference]" >> mem_log.txt
  free -m >> mem_log.txt
  
  python3 infer.py --mode process_cold >> mem_log.txt 2>&1
  
  echo "[After Inference]" >> mem_log.txt
  free -m >> mem_log.txt
done

echo "==================================" >> mem_log.txt
