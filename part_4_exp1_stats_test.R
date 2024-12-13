# require(lme4)
# library(lmerTest)

require("mgcv")


run_stats_test <- function(data_dir) {
    data <- read.csv(paste(data_dir, "/regression_input_data.csv", sep=""))
    data["p_reform"] <- data["p.reform."]

    data["metalinguistic"] <- ifelse(data["way_of_asking"] != "direct", 1, 0)
    data["refer"] <- ifelse(data["way_of_asking"] == "likely_refer" | data["way_of_asking"] == "best_refer", 1, 0)
    data["best"] <- ifelse(data["way_of_asking"] == "best_complete" | data["way_of_asking"] == "best_refer", 1, 0)

    data["choices"] <- ifelse(data["context"] == "choices-all-terms" | data["context"] == "choices-pronoun", 1, 0)
    data["individual_declaration"] <- ifelse(data["context"] == "individual-declaration", 1, 0)
    data["ideology_declaration"] <- ifelse(data["context"] == "ideology-declaration", 1, 0)
    data["null_context"] <- ifelse(data["context"] == "null_context", 1, 0)

    data$name <- as.factor(data$name)
    data$item <- as.factor(data$item)

    # Main regression test
    model <- gam(
        p_reform ~ metalinguistic + refer + best + choices + individual_declaration + ideology_declaration + s(item, bs = 're')  + s(name, bs = 're'),
        family=betar(link="logit"),
        data=data)

    sink(paste(data_dir, "/regression_results.txt", sep=""))
    print(summary(model))
    sink()
    write.csv(summary(model)$p.table, paste(data_dir, "/regression_results.csv", sep=""))

    # # Metalinguistic-only test
    # model_metalinguistic_only <- gam(
    #     p_reform ~ metalinguistic + s(item, bs = 're')  + s(name, bs = 're'),
    #     family=betar(link="logit"),
    #     data=data)

    # sink(paste(data_dir, "/regression_results-metalinguistic-only.txt", sep=""))
    # print(summary(model_metalinguistic_only))
    # sink()
    # write.csv(summary(model_metalinguistic_only)$p.table, paste(data_dir, "/regression_results-metalinguistic.csv", sep=""))

}

models <- c(
    "text-curie-001", "text-davinci-002", "text-davinci-003",
    "flan-t5-small", "flan-t5-large", "flan-t5-xl",
    "llama-2-7B", "llama-3-8B", "llama-3.1-8B")
domains <- c(
    "singular-pronouns-full",
    "role-nouns-full-minus-anchor-flight-attendant",
    "role-nouns-full-minus-anchor-flight-attendant-plus-expanded")
for (model_name in models) {
    for (domain in domains) {
        config_path <- sprintf("analyses/experiment1/%s/%s", domain, model_name)
        print(config_path)
        run_stats_test(config_path)
    }
}
# run_stats_test("analyses/experiment1/role-nouns-full/text-davinci-003")
# run_stats_test("analyses/experiment1/singular-pronouns-full/text-davinci-003")

# run_stats_test("analyses/experiment1/role-nouns-full/text-davinci-002")
# run_stats_test("analyses/experiment1/singular-pronouns-full/text-davinci-002")

# run_stats_test("analyses/experiment1/role-nouns-full/text-curie-001")
# run_stats_test("analyses/experiment1/singular-pronouns-full/text-curie-001")

# run_stats_test("analyses/experiment1/role-nouns-full/flan-t5-xl")
# run_stats_test("analyses/experiment1/singular-pronouns-full/flan-t5-xl")

# run_stats_test("analyses/experiment1/role-nouns-full/flan-t5-large")
# run_stats_test("analyses/experiment1/singular-pronouns-full/flan-t5-large")

# run_stats_test("analyses/experiment1/role-nouns-full/flan-t5-small")
# run_stats_test("analyses/experiment1/singular-pronouns-full/flan-t5-small")