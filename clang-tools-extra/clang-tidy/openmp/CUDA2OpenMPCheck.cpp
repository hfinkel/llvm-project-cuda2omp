//===--- CUDA2OpenMPCheck.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CUDA2OpenMPCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "../utils/LexerUtils.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace openmp {

void CUDA2OpenMPCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(allOf(unless(anyOf(isExpansionInSystemHeader(),
                                                     isImplicit(),
						     // Don't match template
						     // instantiations, we'll
						     // directly convert the
						     // templated kernels.
                                                     ast_matchers::isTemplateInstantiation())),
                                        anyOf(hasAttr(clang::attr::CUDAGlobal),
                                              hasAttr(clang::attr::CUDADevice))))
    .bind("cuda_decl"), this);

  Finder->addMatcher(cudaKernelCallExpr()
    .bind("cuda_call"), this);

  Finder->addMatcher(callExpr(callee(functionDecl(hasName("::cudaDeviceReset"))))
    .bind("cuda_call_device_reset"), this);

  Finder->addMatcher(callExpr(callee(functionDecl(hasName("::cudaDeviceSynchronize"))))
    .bind("cuda_call_device_sync"), this);
}

void CUDA2OpenMPCheck::check(const MatchFinder::MatchResult &Result) {
  const SourceManager &SM = *Result.SourceManager;
  const LangOptions LangOpts = getLangOpts();

  if (const auto *MatchedCall =
      Result.Nodes.getNodeAs<CallExpr>("cuda_call_device_reset")) {

    diag(MatchedCall->getBeginLoc(), "Remove call to CUDA runtime function");
    diag(MatchedCall->getBeginLoc(), "remove function call", DiagnosticIDs::Note)
      << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
                                    MatchedCall->getBeginLoc(),
                                    MatchedCall->getEndLoc()));
    return;
  }

  if (const auto *MatchedCall =
      Result.Nodes.getNodeAs<CallExpr>("cuda_call_device_sync")) {

    diag(MatchedCall->getBeginLoc(), "Remove call to CUDA runtime function");
    diag(MatchedCall->getBeginLoc(), "remove function call", DiagnosticIDs::Note)
      << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
                                    MatchedCall->getBeginLoc(),
                                    MatchedCall->getEndLoc()));
    return;
  }

  if (const auto *MatchedCall =
      Result.Nodes.getNodeAs<CUDAKernelCallExpr>("cuda_call")) {

    auto ConfigBeginSrcLoc = MatchedCall->getConfig()->getBeginLoc();
    auto ConfigEndSrcLoc = MatchedCall->getConfig()->getEndLoc();
    auto LParenSrcLoc =
      utils::lexer::findNextTokenSkippingComments(
        ConfigEndSrcLoc.getLocWithOffset(2), SM, LangOpts)->getLocation();

    diag(MatchedCall->getBeginLoc(), "Remove the CUDA kernel call configuration start markers");
    diag(MatchedCall->getBeginLoc(), "remove '<<<'", DiagnosticIDs::Note)
      << FixItHint::CreateReplacement(CharSourceRange::getTokenRange(
                                        ConfigBeginSrcLoc,
                                        ConfigBeginSrcLoc.getLocWithOffset(2)),
                                      "(");

    diag(MatchedCall->getBeginLoc(), "Remove the CUDA kernel call configuration end markers");
    diag(MatchedCall->getBeginLoc(), "remove '>>>'", DiagnosticIDs::Note)
      << FixItHint::CreateReplacement(CharSourceRange::getTokenRange(
                                        ConfigEndSrcLoc,
                                        LParenSrcLoc),
                                      ", ");

    return;
  }

  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("cuda_decl");

  static bool DidTopInsert = false;
  if (!DidTopInsert) {
    auto StartOfFile = SM.getLocForStartOfFile(SM.getFileID(MatchedDecl->getLocation()));
    diag(MatchedDecl->getLocation(), "Add OpenMP header include and other definitions");
    diag(MatchedDecl->getLocation(), "insert top-level info", DiagnosticIDs::Note)
      << FixItHint::CreateInsertion(StartOfFile,
                                    "#ifdef _OPENMP\n#include <omp.h>\n#endif // _OPENMP\n"
                                    "#ifndef __dim3_defined\n"
                                    "#define __dim3_defined\n"
                                    "struct dim3 {\n"
                                    "unsigned x, y, z;\n"
                                    "dim3(unsigned x, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}\n"
                                    "};\n"
                                    "namespace { const int warpSize = 32; }\n"
                                    "#endif // __dim3_defined\n");

    DidTopInsert = true;
  }

  auto GetAttrReplRange = [&](SourceLocation AttrLoc) {
    // __global__ is generally a macro expansion (in terms of the macro
    // __location__, etc.), so we need the location of the macro name itself.
    if (AttrLoc.isMacroID()) {
      StringRef MacroName;
      while (AttrLoc.isValid() && AttrLoc.isMacroID()) {
        MacroName = Lexer::getImmediateMacroName(AttrLoc, SM, LangOpts);
        auto Loc = SM.getImmediateMacroCallerLoc(AttrLoc);
        if (!Loc.isValid())
          break;
        AttrLoc = Loc;
      }

      // FIXME: Is this right if the whole thing is in a macro?
      return CharSourceRange::getTokenRange(AttrLoc,
               AttrLoc.getLocWithOffset(MacroName.size()));
    }

    return CharSourceRange::getTokenRange(AttrLoc, AttrLoc);
  };

  if (MatchedDecl->hasAttr<CUDADeviceAttr>()) {
    // FIXME: We need to handle host functions too.
    // FIXME: Maybe we need declare variant for these? declare target?

    SourceLocation DeviceLoc = MatchedDecl->getAttr<CUDADeviceAttr>()->getLocation();
    CharSourceRange DeviceReplacementRange = GetAttrReplRange(DeviceLoc);

    diag(MatchedDecl->getLocation(), "Remove the CUDA location attribute");
    diag(MatchedDecl->getLocation(), "remove '__device__'", DiagnosticIDs::Note)
      << FixItHint::CreateRemoval(DeviceReplacementRange);

    diag(MatchedDecl->getLocation(), "Add additional parameters to match the CUDA dimensions variables");
    diag(MatchedDecl->getLocation(), "insert execution parameters", DiagnosticIDs::Note)
      << FixItHint::CreateInsertion(MatchedDecl->getParametersSourceRange().getBegin(),
                                      "const dim3 &gridDim, const dim3 &blockDim, "
                                      "const dim3 &blockIdx, const dim3 &threadIdx, ");

    return;
  }

  assert(MatchedDecl->hasAttr<CUDAGlobalAttr>() && "Expected CUDA global function");

  SourceLocation GlobalLoc = MatchedDecl->getAttr<CUDAGlobalAttr>()->getLocation();
  CharSourceRange GlobalReplacementRange = GetAttrReplRange(GlobalLoc);

  diag(MatchedDecl->getLocation(), "Remove the CUDA location attribute");
  diag(MatchedDecl->getLocation(), "remove '__global__'", DiagnosticIDs::Note)
    << FixItHint::CreateRemoval(GlobalReplacementRange);

  diag(MatchedDecl->getLocation(), "Add additional parameters to match the CUDA dimensions variables");
  diag(MatchedDecl->getLocation(), "insert execution parameters", DiagnosticIDs::Note)
    << FixItHint::CreateInsertion(MatchedDecl->getParametersSourceRange().getBegin(),
                                    "const dim3 &gridDim, const dim3 &blockDim, ");

  if (!MatchedDecl->hasBody())
    return;

  diag(MatchedDecl->getLocation(), "Add OpenMP loops and other CUDA variables");
  diag(MatchedDecl->getLocation(), "insert OpenMP directives", DiagnosticIDs::Note)
    << FixItHint::CreateInsertion(MatchedDecl->getBody()->getBeginLoc().getLocWithOffset(1),
                                  "\n#pragma omp target teams distribute collapse(3)\n"
                                  "for (unsigned __grid_x = 0; __grid_x < gridDim.x; ++__grid_x)\n"
                                  "for (unsigned __grid_y = 0; __grid_y < gridDim.y; ++__grid_y)\n"
                                  "for (unsigned __grid_z = 0; __grid_z < gridDim.z; ++__grid_z) {\n"
                                  "const dim3 blockIdx(__grid_x, __grid_y, __grid_z);\n"
                                  "#pragma omp parallel for collapse(3) num_threads(warpSize)\n"
                                  "for (unsigned __block_x = 0; __block_x < blockDim.x; ++__block_x)\n"
                                  "for (unsigned __block_y = 0; __block_y < blockDim.y; ++__block_y)\n"
                                  "for (unsigned __block_z = 0; __block_z < blockDim.z; ++__block_z) {\n"
                                  "const dim3 threadIdx(__block_x, __block_y, __block_z);\n");

  diag(MatchedDecl->getLocation(), "Add endings of OpenMP loop bodies");
  diag(MatchedDecl->getLocation(), "insert OpenMP loop endings", DiagnosticIDs::Note)
    << FixItHint::CreateInsertion(MatchedDecl->getBody()->getEndLoc(),
                                  "} // end omp parallel for\n"
                                  "} // end omp teams distribute\n");
 }
} // namespace openmp
} // namespace tidy
} // namespace clang
