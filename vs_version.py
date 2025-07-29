# import os
# import re

# def fix_vs_version_rc_files(root_dir, version_str="1,0,0,19387"):
#     # الأنماط اللي عايزين نعدلها
#     fileversion_pattern = re.compile(r'FILEVERSION\s+[\d,]+')
#     productversion_pattern = re.compile(r'PRODUCTVERSION\s+[\d,]+')

#     for foldername, subfolders, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename == "vs_version.rc":
#                 filepath = os.path.join(foldername, filename)
#                 try:
#                     with open(filepath, 'r', encoding='utf-8') as f:
#                         lines = f.readlines()

#                     changed = False
#                     for i, line in enumerate(lines):
#                         if fileversion_pattern.match(line):
#                             lines[i] = f"FILEVERSION {version_str}\n"
#                             changed = True
#                         elif productversion_pattern.match(line):
#                             lines[i] = f"PRODUCTVERSION {version_str}\n"
#                             changed = True

#                     if changed:
#                         with open(filepath, 'w', encoding='utf-8') as f:
#                             f.writelines(lines)
#                         print(f"Fixed {filepath}")

#                 except Exception as e:
#                     print(f"Failed to process {filepath}: {e}")

# if __name__ == "__main__":
#     import sys
#     root = sys.argv[1] if len(sys.argv) > 1 else "."
#     fix_vs_version_rc_files(root)

# import os

# def search_shared_flag(root_dir):
#     results = []
#     for foldername, subfolders, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename.endswith(('.cmake', '.txt', '.rsp', '.bat', '.sh', '.py', '.in')) or '.' not in filename:
#                 filepath = os.path.join(foldername, filename)
#                 try:
#                     with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
#                         for i, line in enumerate(f, 1):
#                             if '/shared' in line.lower():
#                                 results.append((filepath, i, line.strip()))
#                 except Exception as e:
#                     pass
#     return results

# if __name__ == "__main__":
#     import sys
#     root = sys.argv[1] if len(sys.argv) > 1 else '.'
#     matches = search_shared_flag(root)
#     if matches:
#         print(f"Found '/shared' flag in the following files:")
#         for filepath, lineno, line in matches:
#             print(f"{filepath} (line {lineno}): {line}")
#     else:
#         print("No occurrences of '/shared' found.")
# # import os
# # import re

# # def fix_shared_flags(root_dir):
# #     # الأنماط اللي نبحث عنها في ملفات CMakeLists.txt
# #     patterns = [
# #         r'CMAKE_SHARED_LINKER_FLAGS',
# #         r'CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS',
# #         r'CMAKE_SHARED_LIBRARY_LINK_C_FLAGS',
# #     ]

# #     # استعراض المجلدات والملفات
# #     for foldername, subfolders, filenames in os.walk(root_dir):
# #         for filename in filenames:
# #             if filename == "CMakeLists.txt":
# #                 filepath = os.path.join(foldername, filename)
# #                 with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
# #                     lines = f.readlines()

# #                 modified = False
# #                 new_lines = []
# #                 for line in lines:
# #                     original_line = line
# #                     for pattern in patterns:
# #                         # إذا السطر يحتوي على المتغير المطلوب
# #                         if pattern in line:
# #                             # نحذف أو نعدل /shared لو موجود
# #                             # حذف /shared مع أي مسافة قبلها أو بعدها
# #                             line = re.sub(r'\s*/shared\s*', ' ', line, flags=re.IGNORECASE)
# #                             line = line.replace('/shared', '')  # تأمين إزالة أي حالة
# #                             # تنظيف مسافات زائدة
# #                             line = re.sub(r'\s+', ' ', line).strip() + '\n'
# #                             modified = True
# #                     new_lines.append(line)

# #                 if modified:
# #                     print(f"تم تعديل الملف: {filepath}")
# #                     with open(filepath, 'w', encoding='utf-8') as f:
# #                         f.writelines(new_lines)

# # if __name__ == "__main__":
# #     intel_npu_dir = r"C:\Users\LENOVO\Documents\OpenVINO_Work\openvino_disable_fusion\src\plugins\intel_npu"
# #     fix_shared_flags(intel_npu_dir)
# import os
# import re

# def fix_vs_version_rc_files(root_dir, version_str="1,0,0,19387"):
#     fileversion_pattern = re.compile(r'(FILEVERSION\s+)[\d,]+')
#     productversion_pattern = re.compile(r'(PRODUCTVERSION\s+)[\d,]+')

#     for foldername, subfolders, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename == "vs_version.rc":
#                 filepath = os.path.join(foldername, filename)
#                 try:
#                     with open(filepath, 'r', encoding='utf-8') as f:
#                         lines = f.readlines()

#                     changed = False
#                     for i, line in enumerate(lines):
#                         if fileversion_pattern.search(line):
#                             old_line = lines[i].strip()
#                             lines[i] = fileversion_pattern.sub(lambda m: m.group(1) + version_str, line)
#                             print(f"[FILEVERSION] {old_line} → {lines[i].strip()}")
#                             changed = True
#                         elif productversion_pattern.search(line):
#                             old_line = lines[i].strip()
#                             lines[i] = productversion_pattern.sub(lambda m: m.group(1) + version_str, line)
#                             print(f"[PRODUCTVERSION] {old_line} → {lines[i].strip()}")
#                             changed = True

#                     if changed:
#                         with open(filepath, 'w', encoding='utf-8') as f:
#                             f.writelines(lines)
#                         print(f"✅ Fixed {filepath}\n")
#                     else:
#                         print(f"⏩ No changes needed in {filepath}")

#                 except Exception as e:
#                     print(f"❌ Failed to process {filepath}: {e}")

# if __name__ == "__main__":
#     import sys
#     root = sys.argv[1] if len(sys.argv) > 1 else "."
# #     fix_vs_version_rc_files(root)
# import os
# import re

# def fix_vs_version_rc_files(root_dir, version_str="1,0,0,19387"):
#     fileversion_pattern  = re.compile(r'FILEVERSION\s+[\d,]+')
#     productversion_pattern = re.compile(r'PRODUCTVERSION\s+[\d,]+')

#     for foldername, _, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename == "vs_version.rc":
#                 filepath = os.path.join(foldername, filename)
#                 try:
#                     with open(filepath, 'r', encoding='utf-8') as f:
#                         lines = f.readlines()

#                     changed = False
#                     for i, line in enumerate(lines):
#                         if fileversion_pattern.match(line):
#                             lines[i] = f"FILEVERSION {version_str}\n"
#                             changed = True
#                         elif productversion_pattern.match(line):
#                             lines[i] = f"PRODUCTVERSION {version_str}\n"
#                             changed = True

#                     if changed:
#                         with open(filepath, 'w', encoding='utf-8') as f:
#                             f.writelines(lines)
#                         print(f"✅ Fixed {filepath}")

#                 except Exception as e:
#                     print(f"❌ Failed to process {filepath}: {e}")

# if __name__ == "__main__":
#     import sys
#     root = sys.argv[1] if len(sys.argv) > 1 else "."
#     fix_vs_version_rc_files(root)
import os
import re

def fix_vs_version_rc_files(root_dir, version_str="1,0,0,19387"):
    fileversion_pattern  = re.compile(r'FILEVERSION\s+[\d,]+')
    productversion_pattern = re.compile(r'PRODUCTVERSION\s+[\d,]+')

    fixed_files_count = 0

    for foldername, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "vs_version.rc":
                filepath = os.path.join(foldername, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    changed = False
                    for i, line in enumerate(lines):
                        if fileversion_pattern.match(line.strip()):
                            lines[i] = f"FILEVERSION {version_str}\n"
                            changed = True
                        elif productversion_pattern.match(line.strip()):
                            lines[i] = f"PRODUCTVERSION {version_str}\n"
                            changed = True

                    if changed:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        print(f"✅ Fixed {filepath}")
                        fixed_files_count += 1

                except Exception as e:
                    print(f"❌ Failed to process {filepath}: {e}")

    if fixed_files_count == 0:
        print("No files were fixed.")

if __name__ == "__main__":
    import sys
    # افتراضياً المسار داخل فولدر build بالمشروع، تقدر تغيره حسب مكانك
    default_build_path = r"C:\Users\LENOVO\Documents\OpenVINO_Work\openvino_disable_fusion\build"
    root = sys.argv[1] if len(sys.argv) > 1 else default_build_path
    fix_vs_version_rc_files(root)
