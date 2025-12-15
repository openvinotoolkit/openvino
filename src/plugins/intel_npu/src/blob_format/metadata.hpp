
void write_metadata(std::ostream& stream) {
    constexpr std::string_view MAGIC_BYTES = "OVNPU";
    constexpr uint32_t METADATA_VERSION = 0x30000; // 3.0
    OpenvinoVersion ovVersion(OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR, OPENVINO_VERSION_PATCH);

    stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size());
    stream.write(reinterpret_cast<const char*>(&METADATA_VERSION), sizeof(METADATA_VERSION));
    ovVersion.write(stream);
}

bool read_and_validate_metadata(std::istream& stream) {
    constexpr std::string_view MAGIC_BYTES = "OVNPU";
    constexpr uint32_t METADATA_VERSION = 0x30000; // 3.0
    OpenvinoVersion ovVersion(OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR, OPENVINO_VERSION_PATCH);

    std::string magic_read;
    uint32_t meta_version_read;
    
    stream.read(reinterpret_cast<char*>(&magic_read), 5);
    if (magic_read != MAGIC_BYTES) {
        std::cout << "bad magic\n";
        return false;
    }

    stream.read(reinterpret_cast<char*>(&meta_version_read), sizeof(meta_version_read));
    if (meta_version_read != METADATA_VERSION) {
        std::cout << "bad metadata version\n";
        return false;
    }

    OpenvinoVersion ov_version_read(1, 1, 1); // dummy values since there is no default constructor
    ov_version_read.read(stream);

    if (ov_version_read != ovVersion) {
        std::cout << "bad ov version\n";
        return false;
    }
    return true;
}