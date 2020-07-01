/*******************************************************************************
    Copyright (c) The Taichi Authors (2020- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/util/line_appender.h"
#include "taichi/util/testing.h"

TI_NAMESPACE_BEGIN

TI_TEST("LineAppender") {
  SECTION("basic") {
    LineAppender la;
    la.append("1");
    la.append("2");
    la.append("3");

    TI_CHECK(la.lines(/*sep=*/"") == "123");

    la.clear_lines();
    TI_CHECK(la.lines(/*sep=*/"") == "");
  };

  SECTION("cursors1") {
    LineAppender la;
    la.append("1");
    auto c1 = la.make_cursor();
    la.append("4");
    la.append("5");
    auto c2 = la.make_cursor();
    la.append("7");
    la.append("8");
    la.rewind_to_cursor(c1);
    la.append("2");
    la.append("3");
    la.rewind_to_cursor(c2);
    la.append("6");
    la.rewind_to_end();
    la.append("9");

    TI_CHECK(la.lines("") == "123456789");
  };

  SECTION("cursors2") {
    LineAppender la;
    la.append("1");
    auto c1 = la.make_cursor();
    la.append("7");
    la.append("8");
    la.rewind_to_cursor(c1);
    la.append("2");
    la.append("3");
    auto c2 = la.make_cursor();
    la.append("6");
    la.rewind_to_end();
    la.append("9");
    la.rewind_to_cursor(c2);
    la.append("4");
    la.append("5");

    TI_CHECK(la.lines("") == "123456789");
  };

  SECTION("scoped cursor") {
    LineAppender la;
    la.append("1");
    auto c1 = la.make_cursor();
    la.append("4");
    la.append("5");
    {
      ScopedCursor s(la, c1);
      la.append("2");
      la.append("3");
    }
    la.append("6");
    la.append("7");
    
    TI_CHECK(la.lines("") == "1234567");
  };
}

TI_NAMESPACE_END
