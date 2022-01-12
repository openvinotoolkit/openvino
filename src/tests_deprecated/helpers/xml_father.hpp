// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <list>
#include <sstream>
#include <memory>

namespace CommonTestUtils {

template<typename Type>
inline std::string convert2string(Type &input) {
    std::ostringstream convertStream;
    convertStream << input;
    return convertStream.str();
}

#if defined(_MSC_VER) && _MSC_VER > 1800
inline std::string operator ""_s(const char * str, std::size_t len) {
    return std::string(str, str + len);
}

inline std::string make_content(const std::string & tag, const std::string & content) {
    return"<"_s + tag + ">" + content + "</" + tag + ">";
}

inline std::string make_content(const std::string & tag, const std::string & attribute, const std::string & content) {
    return attribute.empty() ? make_content(tag, content) :
        ("<"_s + tag + attribute + ">" + content + "</" + tag + ">");
}

#else

inline std::string make_content(const std::string &tag, const std::string &content) {
    return std::string("<") + tag + ">" + content + "</" + tag + ">\n";
}

inline std::string make_content(const std::string &tag, const std::string &attribute, const std::string &content) {
    return attribute.empty() ? make_content(tag, content) :
           (std::string("<") + tag + attribute + ">" + content + "</" + tag + ">\n");
}

#endif

class ConstCharMaker {
    std::string ref;
public:
    explicit ConstCharMaker(const std::string &ref) : ref(ref) {}

    operator const char *() {
        return ref.c_str();
    }
};

template<class T>
class Token {
    template<class S>
    friend
    class Token;

    T *_f;
    mutable std::string _content;
    std::string _tag;
    std::string _attr;
    mutable std::shared_ptr<Token<Token<T>>> lastTag;

public:
    Token(T *f, const std::string &tag) : _f(f), _tag(tag) {}

    T &close() const {
        return const_cast<T &>(_f->closeToken());
    }

    // direct node content description - without subnodes
    template<class ...Args>
    Token<T> &node(const std::string &name, Args ... content) {
        _content += make_content(name, merge({convert2string(content)...}));
        return *this;
    }

    // combine call to close and creation of new node
    template<class ...Args>
    Token<T> &newnode(const std::string &name, Args ... content) {
        return close().node(name, content ...);
    }


    std::string tag() const {
        return _tag;
    }

    std::string content() const {
        return _content;
    }

    void add_content(std::string content) {
        _content += content;
    }

    std::string attr() const {
        return _attr;
    }

    template<typename Arg>
    Token<T> &attr(const std::string &attributeType, const Arg &attribute) {
        _attr = merge({_attr, attributeType + "=\"" + convert2string(attribute) + "\""});
        return *this;
    }

    operator std::string() {
        return closeAll().please();
    }

    ConstCharMaker c_str() const {
        return ConstCharMaker(closeAll().please());
    }

    Token<Token<T>> &node(const std::string &tag) {
        lastTag = std::make_shared<Token<Token<T>>>(this, tag);
        return *lastTag;
    }


private :
    auto closeAll() const -> decltype(_f->closeAll()) {
        closeToken();
        return _f->closeAll();
    }

    std::string please() {
        if (lastTag.get() != nullptr) {
            lastTag->close();
        }

        return _content;
    }

    const Token<T> &closeToken() const {
        if (lastTag) {
            _content += make_content(lastTag->tag(), lastTag->attr(), lastTag->content());
            lastTag.reset();
        }
        return *this;
    }

    std::string merge(std::initializer_list<std::string> strList) const {
        std::stringstream ret;

        for (auto it = strList.begin(); it != strList.end(); it++) {
            ret << *it;
            if ((it + 1) != strList.end()) {
                ret << " ";
            }
        }
        return ret.str();
    }
};

class XMLFather {
    friend class Token<XMLFather>;

    std::list<std::string> tokens;
    std::shared_ptr<Token<XMLFather>> lastTag;
    std::string _please;

public:
    static XMLFather make_without_schema() {
        auto x = XMLFather();
        x.tokens.clear();
        return x;
    }

    XMLFather() {
        tokens.push_back("<?xml version=\"1.0\"?>\n");
    }

    Token<XMLFather> &node(const std::string &tag) {
        lastTag = std::make_shared<Token<XMLFather>>(this, tag);
        return *lastTag;
    }

    std::string please() {
        if (!_please.empty()) {
            return _please;
        }
        if (lastTag.get() != nullptr) {
            lastTag->close();
        }
        std::stringstream ss;
        for (auto s : tokens) {
            ss << s;
        }
        return _please = ss.str();
    }

    operator std::string() {
        return please();
    }

protected:
    XMLFather &closeAll() {
        return closeToken();
    }

    XMLFather &closeToken() {
        tokens.push_back(make_content(lastTag->tag(), lastTag->attr(), lastTag->content()));
        lastTag.reset();
        return *this;
    }
};

}  // namespace CommonTestUtils
