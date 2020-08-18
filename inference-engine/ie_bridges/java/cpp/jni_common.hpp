#include <jni.h>   // JNI header provided by JDK
#include <stdio.h> // C Standard IO Header

static void throwJavaException(JNIEnv *env, const std::exception *e, const char *method)
{
    std::string what = "unknown exception";
    jclass je = 0;

    if(e){
        std::string exception_type = "InferenceEngineException";
        what = '\n' + exception_type + ": " + '\n' + '\t' + std::string(method) + ": " + e->what();
    }
    
    if (!je)
        je = env->FindClass("java/lang/Exception");

    env->ThrowNew(je, what.c_str());

    (void)method;
}

static std::string jstringToString(JNIEnv *env, jstring jstr)
{
    static const char method_name[] = "jstringToString";  
    try
    {
        const char *utf_str = env->GetStringUTFChars(jstr, 0);
        std::string n_str(utf_str ? utf_str : "");
        env->ReleaseStringUTFChars(jstr, utf_str);
        return n_str;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return nullptr;
}

static std::map<std::string, std::string> javaMapToMap(JNIEnv *env, jobject java_map)
{
    static const char method_name[] = "javaMapToMap";
    try
    {
        jclass map_class = env->GetObjectClass(java_map);
        jmethodID entrySet_method_id = env->GetMethodID(map_class, "entrySet", "()Ljava/util/Set;");
        jobject entry_set = env->CallObjectMethod(java_map, entrySet_method_id);
        
        jclass set_class = env->FindClass("java/util/Set");
        jmethodID iterator_method_id = env->GetMethodID(set_class, "iterator", "()Ljava/util/Iterator;");
        jobject iterator = env->CallObjectMethod(entry_set, iterator_method_id);
        
        jclass iterator_class = env->FindClass("java/util/Iterator");
        jmethodID hasNext_method_id = env->GetMethodID(iterator_class, "hasNext", "()Z");
        jmethodID next_method_id = env->GetMethodID(iterator_class, "next", "()Ljava/lang/Object;");
        bool hasNext = (bool) (env->CallBooleanMethod(iterator, hasNext_method_id) == JNI_TRUE);

        jclass mapentry_class = (env)->FindClass("java/util/Map$Entry");
        jmethodID get_key_method_id = env->GetMethodID(mapentry_class, "getKey", "()Ljava/lang/Object;");
        jmethodID get_value_method_id = env->GetMethodID(mapentry_class, "getValue", "()Ljava/lang/Object;");

        jclass string_class = env->FindClass("java/lang/String");
        jmethodID to_string_method_id = env->GetMethodID(string_class, "toString", "()Ljava/lang/String;");

        std::map<std::string, std::string> c_map;

        while(hasNext) {
		    jobject entry = env->CallObjectMethod(iterator, next_method_id);

		    jstring key = (jstring) env->CallObjectMethod(env->CallObjectMethod(entry, get_key_method_id), to_string_method_id);
		    jstring value = (jstring) env->CallObjectMethod(env->CallObjectMethod(entry, get_value_method_id), to_string_method_id);    

            c_map.insert(std::make_pair(jstringToString(env, key), jstringToString(env, value)));

		    hasNext = (bool) (env->CallBooleanMethod(iterator, hasNext_method_id) == JNI_TRUE);
	    }

        return c_map;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return std::map<std::string, std::string>();
}

static std::map<std::string, std::vector<size_t>> javaMapToMap_1(JNIEnv *env, jobject java_map)
{
    static const char method_name[] = "javaMapToMap";
    try
    {
        jclass map_class = env->GetObjectClass(java_map);
        jmethodID entrySet_method_id = env->GetMethodID(map_class, "entrySet", "()Ljava/util/Set;");
        jobject entry_set = env->CallObjectMethod(java_map, entrySet_method_id);
        
        jclass set_class = env->FindClass("java/util/Set");
        jmethodID iterator_method_id = env->GetMethodID(set_class, "iterator", "()Ljava/util/Iterator;");
        jobject iterator = env->CallObjectMethod(entry_set, iterator_method_id);
        
        jclass iterator_class = env->FindClass("java/util/Iterator");
        jmethodID hasNext_method_id = env->GetMethodID(iterator_class, "hasNext", "()Z");
        jmethodID next_method_id = env->GetMethodID(iterator_class, "next", "()Ljava/lang/Object;");
        bool hasNext = (bool) (env->CallBooleanMethod(iterator, hasNext_method_id) == JNI_TRUE);

        jclass mapentry_class = (env)->FindClass("java/util/Map$Entry");
        jmethodID get_key_method_id = env->GetMethodID(mapentry_class, "getKey", "()Ljava/lang/Object;");
        jmethodID get_value_method_id = env->GetMethodID(mapentry_class, "getValue", "()Ljava/lang/Object;");

        jclass string_class = env->FindClass("java/lang/String");
        jmethodID to_string_method_id = env->GetMethodID(string_class, "toString", "()Ljava/lang/String;");

        std::map<std::string, std::vector<size_t>> c_map;

        while(hasNext) {
		    jobject entry = env->CallObjectMethod(iterator, next_method_id);

		    jstring key = (jstring)env->CallObjectMethod(env->CallObjectMethod(entry, get_key_method_id), to_string_method_id);
            jintArray value = (jintArray)env->CallObjectMethod(entry, get_value_method_id); 

            const jsize length = env->GetArrayLength(value);
            jint *data = env->GetIntArrayElements(value, 0);
            
            c_map.insert(std::make_pair(jstringToString(env, key), std::vector<size_t>(data, data + length)));

            env->ReleaseIntArrayElements(value, data, 0);

		    hasNext = (bool)(env->CallBooleanMethod(iterator, hasNext_method_id) == JNI_TRUE);
	    }
        
        return c_map;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }
    return std::map<std::string, std::vector<size_t>>();
}

template<typename Key, typename Val>
static Key find_by_value(const std::map<Key, Val>& map, Val& value) 
{
    for (auto &itr : map)
        if (itr.second == value)
            return itr.first;
            
    throw std::runtime_error("No such value in java bindings enum!");
}

static std::vector<size_t> jintArrayToVector(JNIEnv *env, jintArray dims)
{
    static const char method_name[] = "jintArrayToVector";
    try
    {
        const jsize length = env->GetArrayLength(dims);
        jint *data = env->GetIntArrayElements(dims, 0);

        std::vector<size_t> res(data, data + length);
        env->ReleaseIntArrayElements(dims, data, 0);

        return res;
    }
    catch (const std::exception &e)
    {
        throwJavaException(env, &e, method_name);
    }
    catch (...)
    {
        throwJavaException(env, 0, method_name);
    }

    return std::vector<size_t>();
}
