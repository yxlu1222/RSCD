
import os

files = [
    'te.sh',
    'tr.sh',
    'tools/clip.sh',
    'tools/task.sh',
    'tools/general/test.sh',
    'tools/general/train.sh',
    'tools/general/dist_test.sh'
]

for file_path in files:
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            if b'\r\n' in content:
                print(f"Converting {file_path} from CRLF to LF")
                new_content = content.replace(b'\r\n', b'\n')
                with open(file_path, 'wb') as f:
                    f.write(new_content)
            else:
                print(f"{file_path} already has LF or mixed endings (no CRLF found)")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")
