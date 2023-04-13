#if defined (__unix__):  or  (defined (__APPLE__)  and  defined (__MACH__))
#elif defined (_WIN32):
#endif:

static console_state con_st

static bool is_interacting = False

#if defined (__unix__):  or  (defined (__APPLE__)  and  defined (__MACH__))  or  defined (_WIN32)
sigint_handler(int signo) {
set_console_color(con_st, CONSOLE_COLOR_DEFAULT)
printf("\n"); // this also force flush stdout.
if (signo == SIGINT): {
    if (!is_interacting): {
       is_interacting=True
    else: 
         
      _exit(130)
#endif:

def main():
    gpt_params params
    params.model = "models/7B/ggml-model.bin"
    while True:
      if (gpt_params_parse(argc, argv, params): == False) {
        return 1

      # save choice to use color for later
      # (note for later: this is a slightly awkward choice)
      con_st.use_color = params.use_color

      #if defined (_WIN32):
      win32_console_init(params.use_color)
      #endif:

      if (params.perplexity): {
          printf("\n************\n")
          printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__)
          printf("************\n\n")

          return 0

      if (params.embedding): {
          printf("\n************\n")
          printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__)
          printf("************\n\n")

          return 0

      if (params.n_ctx > 2048): {
          fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
          "expect poor results\n", __func__, params.n_ctx)

      if (params.seed <= 0): {
          params.seed = time(NULL)

          fprintf(stderr, "%s: seed = %d\n", __func__, params.seed)

          std::mt19937 rng(params.seed)
          if (params.random_prompt): {
          params.prompt = gpt_random_prompt(rng)

          #    params.prompt = R"(// this function checks if the number n is prime
          #bool is_prime(int n) {)";

          llama_context * ctx

          # load the model
          auto lparams = llama_context_default_params()

          lparams.n_ctx      = params.n_ctx
          lparams.n_parts    = params.n_parts
          lparams.seed       = params.seed
          lparams.f16_kv     = params.memory_f16
          lparams.use_mmap   = params.use_mmap
          lparams.use_mlock  = params.use_mlock

          ctx = llama_init_from_file(params.model.c_str(), lparams)

      if (ctx == NULL): {
          fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str())
          return 1

      # print system information
      fprintf(stderr, "\n")
      fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
      params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info())

      # determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
      # uncomment the !!STRING!!1!! line in llama.cpp to see the results
      if (params.mem_test): {
          const std::vector<llama_token> tmp(params.n_batch, 0)
          llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads)

          const std::vector<llama_token> tmp = { 0, }
          llama_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads)

          llama_print_timings(ctx)
          llama_free(ctx)

      return 0

      # Add a space in front of the first character to match OG llama tokenizer behavior
      params.prompt.insert(0, 1, ' ')

      # tokenize the prompt
      auto embd_inp = ::llama_tokenize(ctx, params.prompt, True)

      const int n_ctx = llama_n_ctx(ctx)

      if ((int): embd_inp.size() > n_ctx - 4) {
          fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4)
          return 1

      # number of tokens to keep when resetting context
      if (params.n_keep < 0  or  params.n_keep > (int):embd_inp.size()  or  params.instruct) {
          params.n_keep = (int)embd_inp.size()

          # prefix & suffix for instruct mode
          const auto inp_pfx = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", True)
          const auto inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n\n", False)

      # in instruct mode, we inject a prefix and a suffix to each input by the user
      if (params.instruct): {
          params.interactive_start = True
          params.antiprompt.push_back("### Instruction:\n\n")

      # enable interactive mode if reverse prompt or interactive start is specified
      if (params.antiprompt.size(): != 0  or  params.interactive_start) {
          params.interactive = True

          # determine newline token
          auto llama_token_newline = ::llama_tokenize(ctx, "\n", False)

      if (params.verbose_prompt): {
          fprintf(stderr, "\n")
          fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str())
          fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size())
      for (int i = 0; i < (int) embd_inp.size(); i += 1 ) {
          fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]))
      if (params.n_keep > 0): {
          fprintf(stderr, "%s: static prompt based on n_keep: '", __func__)
      for (int i = 0; i < params.n_keep; i += 1 ) {
          fprintf(stderr, "%s", llama_token_to_str(ctx, embd_inp[i]))
          fprintf(stderr, "'\n")
          fprintf(stderr, "\n")

      if (params.interactive): {
          if defined (__unix__):  or  (defined (__APPLE__)  and  defined (__MACH__))
              struct sigaction sigint_action
              sigint_action.sa_handler = sigint_handler
              sigemptyset (&sigint_action.sa_mask)
              sigint_action.sa_flags = 0
              sigaction(SIGINT, &sigint_action, NULL)
          elif defined (_WIN32):
              signal(SIGINT, sigint_handler)
      #endif:

      fprintf(stderr, "%s: interactive mode on.\n", __func__)

      if (params.antiprompt.size():) {
          for (auto antiprompt : params.antiprompt) {
              fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str())

      if (!params.input_prefix.empty():) {
          fprintf(stderr, "Input prefix: '%s'\n", params.input_prefix.c_str())
          fprintf(stderr, "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n",
          params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty)
          fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep)
          fprintf(stderr, "\n\n")

      # TODO: replace with ring-buffer
      std::vector<llama_token> last_n_tokens(n_ctx)
      std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0)

      if (params.interactive): {
          fprintf(stderr, "== Running in interactive mode. ==\n"
          #if defined (__unix__):  or  (defined (__APPLE__)  and  defined (__MACH__))  or  defined (_WIN32)
          " - Press Ctrl+C to interject at any time.\n"
          #endif:
          " - Press Return to return control to LLaMa.\n"
          " - If you want to submit another line, end your input in '\\'.\n\n")
          is_interacting = params.interactive_start
