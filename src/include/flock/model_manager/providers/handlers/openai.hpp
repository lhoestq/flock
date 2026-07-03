#pragma once

#include "flock/model_manager/providers/handlers/base_handler.hpp"
#include "session.hpp"
#include <cstdlib>
#include <sstream>

namespace flock {

class OpenAIModelManager : public BaseModelProviderHandler {
public:
    OpenAIModelManager(std::string token, std::string api_base_url, bool throw_exception)
        : BaseModelProviderHandler(throw_exception), _token(token), _session("OpenAI", throw_exception) {
        _session.setToken(token, "");
        if (api_base_url.empty()) {
            _api_base_url = "https://api.openai.com/v1/";
        } else {
            _api_base_url = api_base_url + '/';
        }
        _session.setUrl(_api_base_url);
    }

    OpenAIModelManager(const OpenAIModelManager&) = delete;
    OpenAIModelManager& operator=(const OpenAIModelManager&) = delete;
    OpenAIModelManager(OpenAIModelManager&&) = delete;
    OpenAIModelManager& operator=(OpenAIModelManager&&) = delete;

protected:
    std::string _token;
    std::string _api_base_url;
    Session _session;

    std::string getCompletionUrl() const override {
        return _api_base_url + "chat/completions";
    }
    std::string getEmbedUrl() const override {
        return _api_base_url + "embeddings";
    }
    std::string getTranscriptionUrl() const override {
        return _api_base_url + "audio/transcriptions";
    }
    void prepareSessionForRequest(const std::string& url) override {
        _session.setUrl(url);
    }
    void setParameters(const std::string& data, const std::string& contentType = "") override {
        if (contentType != "multipart/form-data") {
            _session.setBody(data);
        }
    }
    auto postRequest(const std::string& contentType) -> decltype(((Session*) nullptr)->postPrepare(contentType)) override {
        return _session.postPrepare(contentType);
    }
    std::vector<std::string> getExtraHeaders() const override {
        return {"Authorization: Bearer " + _token};
    }
    void checkProviderSpecificResponse(const nlohmann::json& response, RequestType request_type) override {
        if (request_type == RequestType::Transcription) {
            return;// No specific checks needed for transcriptions
        }
        bool is_completion = (request_type == RequestType::Completion);
        if (is_completion) {
            if (response.contains("choices") && response["choices"].is_array() && !response["choices"].empty()) {
                const auto& choice = response["choices"][0];
                if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
                    std::string finish_reason = choice["finish_reason"].get<std::string>();
                    if (finish_reason == "length") {
                        throw TokenLimitExceededError();
                    }
                    if (finish_reason != "stop") {
                        throw std::runtime_error("OpenAI API did not finish successfully. finish_reason: " + finish_reason);
                    }
                }
            }
        } else {
            if (response.contains("data") && response["data"].is_array() && response["data"].empty()) {
                throw std::runtime_error("OpenAI API returned empty embedding data.");
            }
        }
    }

    nlohmann::json ExtractCompletionOutput(const nlohmann::json& response) const override {
        if (response.contains("choices") && response["choices"].is_array() && !response["choices"].empty()) {
            const auto& choice = response["choices"][0];
            if (choice.contains("message") && choice["message"].contains("content")) {
                auto content = choice["message"]["content"].get<std::string>();
                try {
                    // Try to parse content as JSON first (structured output)
                    return nlohmann::json::parse(content);
                } catch (const nlohmann::json::parse_error&) {
                    // Not valid JSON - return as string
                    return content;
                }
            }
        }
        return {};
    }

    nlohmann::json ExtractEmbeddingVector(const nlohmann::json& response) const override {
        auto results = nlohmann::json::array();
        if (response.contains("data") && response["data"].is_array() && !response["data"].empty()) {
            const auto& embeddings = response["data"];
            for (const auto& embedding: embeddings) {
                results.push_back(embedding["embedding"]);
            }
        }
        return results;
    }

    std::pair<int64_t, int64_t> ExtractTokenUsage(const nlohmann::json& response) const override {
        int64_t input_tokens = 0;
        int64_t output_tokens = 0;
        if (response.contains("usage") && response["usage"].is_object()) {
            const auto& usage = response["usage"];
            if (usage.contains("prompt_tokens") && usage["prompt_tokens"].is_number()) {
                input_tokens = usage["prompt_tokens"].get<int64_t>();
            }
            if (usage.contains("completion_tokens") && usage["completion_tokens"].is_number()) {
                output_tokens = usage["completion_tokens"].get<int64_t>();
            }
        }
        return {input_tokens, output_tokens};
    }


    nlohmann::json ExtractTranscriptionOutput(const nlohmann::json& response) const override {
        // Transcription API returns JSON with "text" field when response_format=json
        if (response.contains("text") && !response["text"].is_null()) {
            return response["text"].get<std::string>();
        }
        return "";
    }

    // Reconstruct full completion JSON from SSE streaming chunks.
    // OpenAI format: each chunk has {"choices":[{"delta":{"content":"..."}}]}
    // The last chunk has finish_reason and usage.
    nlohmann::json ReconstructFromStreamedChunks(const std::string& sse_raw) const override {
        std::string accumulated_content;
        std::string finish_reason;
        nlohmann::json usage;

        // Parse SSE chunks: each chunk is "data: {json}\n\n"
        std::istringstream stream(sse_raw);
        std::string line;
        std::string data_buffer;

        while (std::getline(stream, line)) {
            // Trim whitespace
            while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.erase(line.begin());
            while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) line.pop_back();

            if (line.rfind("data: ", 0) == 0) {
                std::string json_str = line.substr(6);
                if (json_str.empty()) continue;

                // Skip DONE sentinel
                if (json_str == "[DONE]") continue;

                // Skip invalid JSON
                if (json_str[0] != '{' && json_str[0] != '[') continue;

                nlohmann::json chunk;
                try {
                    chunk = nlohmann::json::parse(json_str);
                } catch (...) {
                    continue;
                }

                // Accumulate delta content from choices.
                // Standard OpenAI: content is in delta.content.
                // vLLM (e.g. Qwen3.x): sends content="" and puts text in delta.reasoning.
                if (chunk.contains("choices") && chunk["choices"].is_array()) {
                    for (const auto& choice: chunk["choices"]) {
                        if (choice.contains("delta") && choice["delta"].is_object()) {
                            auto& delta = choice["delta"];
                            // Prefer delta.content, but fall back to reasoning if content is empty
                            // (vLLM sends content="" and all text in reasoning)
                            if (delta.contains("content") && delta["content"].is_string()) {
                                std::string c = delta["content"].get<std::string>();
                                if (!c.empty()) {
                                    accumulated_content += c;
                                } else if (delta.contains("reasoning") && delta["reasoning"].is_string()) {
                                    accumulated_content += delta["reasoning"].get<std::string>();
                                }
                            } else if (delta.contains("reasoning") && delta["reasoning"].is_string()) {
                                // No content field at all, use reasoning
                                accumulated_content += delta["reasoning"].get<std::string>();
                            }
                        }
                        // Capture finish_reason from the last chunk that has it
                        if (choice.contains("finish_reason") && choice["finish_reason"].is_string()) {
                            finish_reason = choice["finish_reason"].get<std::string>();
                        }
                    }
                }

                // Capture usage - usually in the last chunk (can be at top level)
                if (chunk.contains("usage")) {
                    usage = chunk["usage"];
                }
            }
        }

        // Build the reconstructed JSON in the same shape as non-streaming response
        nlohmann::json choice = {
                {"index", 0},
                {"message", {"role", "assistant", "content", accumulated_content}}
        };
        if (!finish_reason.empty()) choice["finish_reason"] = finish_reason;
        nlohmann::json reconstructed = {
                {"choices", nlohmann::json::array({choice})}
        };

        if (!usage.empty()) {
            reconstructed["usage"] = usage;
        }

        return reconstructed;
    }
};

}// namespace flock
