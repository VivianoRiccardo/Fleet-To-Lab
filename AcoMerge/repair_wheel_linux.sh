mv libezsparkc.so /usr/local/lib/
#sh create_bundle_for_repair_linux.sh
auditwheel repair ./dist/*.whl --plat manylinux2014_x86_64
